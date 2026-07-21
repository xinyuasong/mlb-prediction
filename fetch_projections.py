"""Pull pitcher projections into projections.csv (starter priors).

FanGraphs paywalled the csv export and their frontend API usually 403s
behind Cloudflare, so the reliable path is pasting the projections page
into a text file and running --paste. The parser handles the tab-separated
'# Name Team ...' format (batters + pitchers sections) and resolves names
to MLBAM ids via one MLB API call, accent/suffix insensitive. Unmatched
names go to projections_unmatched.csv - only matters if one's a starter.

    python fetch_projections.py --paste zips_paste.txt
    python fetch_projections.py --type rzips        # API attempt
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

import requests

MLB_API = "https://statsapi.mlb.com/api/v1"
FG_API = "https://www.fangraphs.com/api/projections"


# ----------------------------------------------------------- name matching
def norm_name(s: str) -> str:
    """Accent-, case-, punctuation- and suffix-insensitive key."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return " ".join(s.split())


def mlbam_directory(season: int) -> dict[str, list[tuple[int, str]]]:
    """norm_name -> [(mlbam_id, team_abbrev-ish)] for every MLB player."""
    r = requests.get(f"{MLB_API}/sports/1/players",
                     params={"season": season}, timeout=30)
    r.raise_for_status()
    out: dict[str, list[tuple[int, str]]] = {}
    for p in r.json().get("people", []):
        key = norm_name(p.get("fullName", ""))
        out.setdefault(key, []).append((p["id"], p.get("currentTeam", {}).get("id")))
    return out


# ------------------------------------------------------------ paste parser
def read_text_tolerant(path: Path) -> str:
    """Read a text file whatever its encoding: pastes come from browsers
    and editors on any OS, and accented player names (0xE1 = 'á' in
    cp1252) crash a strict UTF-8 read. latin-1 maps every byte, so the
    final fallback never fails."""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1")  # unreachable, kept for safety


def parse_paste(path: Path) -> list[dict]:
    """Parse a FanGraphs page paste. Returns pitcher rows with Name/Team/
    ERA/FIP. Tab-delimited is expected; header rows locate the columns."""
    rows, cols, in_pitchers = [], None, False
    for line in read_text_tolerant(path).splitlines():
        parts = line.split("\t")
        low = line.strip().lower()
        if low in ("pitchers", "batters"):
            in_pitchers = low == "pitchers"
            cols = None
            continue
        if parts and parts[0] in ("#",):  # header row
            cols = {c.strip(): i for i, c in enumerate(parts)}
            # a pitchers header has IP/ERA; batters has PA/wOBA
            if "IP" in cols and "ERA" in cols:
                in_pitchers = True
            elif "PA" in cols:
                in_pitchers = False
            continue
        if not in_pitchers or cols is None or len(parts) < 5:
            continue
        try:
            rows.append({
                "Name": parts[cols["Name"]].strip(),
                "Team": parts[cols["Team"]].strip(),
                "IP": float(parts[cols["IP"]] or 0),
                "GS": float(parts[cols.get("GS", 0)] or 0) if "GS" in cols else 0.0,
                "ERA": float(parts[cols["ERA"]] or 0),
                "FIP": float(parts[cols["FIP"]] or 0),
            })
        except (KeyError, ValueError, IndexError):
            continue
    return rows


def parse_paste_batters(path: Path) -> list[dict]:
    """Same paste file, batters section (Name/Team/PA/wOBA + slash line)."""
    rows, cols, in_batters = [], None, False
    for line in read_text_tolerant(path).splitlines():
        parts = line.split("\t")
        low = line.strip().lower()
        if low in ("pitchers", "batters"):
            in_batters = low == "batters"
            cols = None
            continue
        if parts and parts[0] == "#":
            cols = {c.strip(): i for i, c in enumerate(parts)}
            in_batters = "PA" in cols and "wOBA" in cols
            continue
        if not in_batters or cols is None or len(parts) < 5:
            continue
        try:
            rows.append({k: parts[cols[k]].strip() for k in
                         ("Name", "Team", "PA", "AVG", "OBP", "SLG", "wOBA", "wRC+")
                         if k in cols})
        except (KeyError, IndexError):
            continue
    return rows


# ------------------------------------------------------------ fangraphs api
def fetch_fangraphs(proj_type: str) -> list[dict] | None:
    try:
        r = requests.get(FG_API,
                         params={"type": proj_type, "stats": "pit", "pos": "all"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code == 403:
            print("FanGraphs API returned 403 (Cloudflare). This is common "
                  "for scripted access - use --paste instead.")
            return None
        r.raise_for_status()
        data = r.json()
        rows = []
        for p in (data if isinstance(data, list) else []):
            mlbam = p.get("xMLBAMID") or p.get("MLBAMID")
            if not mlbam:
                continue
            rows.append({"MLBAMID": int(mlbam),
                         "Name": p.get("PlayerName", ""),
                         "Team": p.get("Team", ""),
                         "ERA": p.get("ERA", ""), "FIP": p.get("FIP", "")})
        return rows or None
    except requests.RequestException as e:
        print(f"FanGraphs API unreachable: {e}")
        return None


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="rzips",
                    help="FanGraphs projection type (rzips, steamerr, ratcdc...)")
    ap.add_argument("--paste", type=Path, default=None,
                    help="fallback: text file pasted from the FanGraphs page")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("projections.csv"))
    ap.add_argument("--batters-out", type=Path, default=Path("zips_batters.csv"),
                    help="also extract the batters section of a paste")
    args = ap.parse_args()

    rows = None
    if not args.paste:
        rows = fetch_fangraphs(args.type)
        if rows is None:
            sys.exit("API path failed and no --paste file given. Copy the "
                     "projections table from fangraphs.com into a text file "
                     "and re-run with --paste that_file.txt")
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["MLBAMID", "Name", "Team", "ERA", "FIP"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} pitcher projections (API) to {args.out}")
        return

    # ---- paste path: parse, then resolve MLBAM ids by name
    pitchers = parse_paste(args.paste)
    if not pitchers:
        sys.exit(f"No pitcher rows parsed from {args.paste} - is it a raw "
                 "tab-separated page paste with the '# Name Team...' header?")
    print(f"Parsed {len(pitchers)} pitcher rows from paste")
    directory = mlbam_directory(args.season)

    matched, unmatched = [], []
    for r in pitchers:
        cands = directory.get(norm_name(r["Name"]), [])
        if len(cands) == 1 or (len(cands) > 1 and r["Name"]):
            # name collisions (e.g. two Max Muncys, Luis Garcias) are rare
            # but real; take the first and note ambiguity
            mlbam = cands[0][0]
            matched.append({"MLBAMID": mlbam, "Name": r["Name"],
                            "Team": r["Team"], "ERA": r["ERA"], "FIP": r["FIP"],
                            "ambiguous": len(cands) > 1})
        else:
            unmatched.append(r)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["MLBAMID", "Name", "Team", "ERA", "FIP"])
        w.writeheader()
        for m in matched:
            w.writerow({k: m[k] for k in ("MLBAMID", "Name", "Team", "ERA", "FIP")})
    amb = sum(1 for m in matched if m["ambiguous"])
    print(f"Wrote {len(matched)} projections to {args.out} "
          f"({amb} ambiguous name matches, first candidate used)")
    if unmatched:
        upath = args.out.with_name("projections_unmatched.csv")
        with open(upath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Name", "Team", "ERA", "FIP", "IP", "GS"])
            w.writeheader()
            w.writerows(unmatched)
        print(f"{len(unmatched)} names not matched to MLBAM ids -> {upath}")
        starters = [u["Name"] for u in unmatched if u.get("GS", 0) >= 3]
        if starters:
            print("  unmatched STARTERS (these matter):", ", ".join(starters[:10]))

    # ---- batters (bonus: stored for future lineup-prior work)
    batters = parse_paste_batters(args.paste)
    if batters:
        bm, bu = 0, 0
        with open(args.batters_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["MLBAMID", "Name", "Team", "PA",
                                              "AVG", "OBP", "SLG", "wOBA", "wRC+"])
            w.writeheader()
            for r in batters:
                cands = directory.get(norm_name(r["Name"]), [])
                if cands:
                    w.writerow({"MLBAMID": cands[0][0], **r})
                    bm += 1
                else:
                    bu += 1
        print(f"Wrote {bm} batter projections to {args.batters_out} "
              f"({bu} unmatched)")


if __name__ == "__main__":
    main()
