"""All external I/O: MLB Stats API, Open-Meteo, The Odds API, csv files.

Everything is keyed by team/player id, retried, and disk-cached when the data
can't change. Fetchers take an as_of date where leakage matters (backtests
must never see the future). Missing data -> None + log line, never a crash.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from . import config

log = logging.getLogger("mlb_model.data")


def parse_ip(ip_str: str) -> float:
    """MLB notates innings as '123.1'/'123.2' meaning 1/3 and 2/3 innings."""
    whole, _, frac = str(ip_str).partition(".")
    return int(whole or 0) + (int(frac) if frac else 0) / 3.0


@dataclass
class PitcherLine:
    """A pitcher's aggregate line over some date window."""
    pitcher_id: int
    name: str
    ip: float
    runs: float          # ALL runs allowed - unearned runs are real runs.
    hand: str | None = None  # 'L'/'R' when known

    @property
    def ra9(self) -> float | None:
        return 9.0 * self.runs / self.ip if self.ip > 0 else None


@dataclass
class TeamSeasonLine:
    """Team totals as of some date (offense + defense)."""
    team_id: int
    name: str
    games: int
    runs_scored: float
    runs_allowed: float
    innings_pitched: float  # defensive innings -> true RA9 (not runs/game)


@dataclass
class BatterSplits:
    """Batter platoon inputs: counting stats vs LHP and vs RHP."""
    player_id: int
    name: str
    bat_side: str                      # 'L', 'R', or 'S'
    vs: dict[str, dict] = field(default_factory=dict)  # 'vl'/'vr' -> stat dict


@dataclass
class Weather:
    temp_f: float
    wind_mph: float
    wind_dir_deg: float   # meteorological: direction wind blows FROM
    source: str           # 'forecast' | 'archive'


@dataclass
class BullpenUsage:
    team_id: int
    relief_ip_3d: float
    relief_pitches_3d: int
    games_counted: int


@dataclass
class GameInfo:
    game_pk: int
    game_date: str
    game_time_utc: str            # ISO timestamp of first pitch
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    home_prob_id: int | None
    away_prob_id: int | None
    home_prob_name: str
    away_prob_name: str
    state: str
    home_score: int | None = None
    away_score: int | None = None
    home_lineup: list[int] = field(default_factory=list)   # batter ids, order
    away_lineup: list[int] = field(default_factory=list)


class MLBDataClient:
    def __init__(self, cache_dir: str | None = config.CACHE_DIR):
        self.s = requests.Session()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ http
    def _get_url(self, url: str, cacheable: bool = False,
                 cache_tag: str = "", **params) -> dict | None:
        """GET with retries. cacheable=True -> disk cache (for data that can
        no longer change). cache_tag distinguishes otherwise-identical
        requests whose answer changes over time (e.g. 'asof-2026-07-19')."""
        key = None
        if cacheable and self.cache_dir:
            raw = (url.split("//", 1)[-1].replace("/", "_") + "_" + cache_tag
                   + "_".join(f"{k}-{v}" for k, v in sorted(params.items())))
            key = self.cache_dir / (raw[:180].replace(":", "") + ".json")
            if key.exists():
                try:
                    return json.loads(key.read_text())
                except (json.JSONDecodeError, OSError):
                    key.unlink(missing_ok=True)
        for attempt in range(1, config.HTTP_RETRIES + 1):
            try:
                r = self.s.get(url, params=params, timeout=config.HTTP_TIMEOUT)
                r.raise_for_status()
                data = r.json()
                if key:
                    key.write_text(json.dumps(data))
                return data
            except requests.RequestException as e:
                log.warning("API error (%s/%s) on %s: %s",
                            attempt, config.HTTP_RETRIES, url, e)
                time.sleep(1.5 * attempt)
        log.error("API failed permanently: %s params=%s", url, params)
        return None

    def _get(self, path: str, cacheable: bool = False,
             cache_tag: str = "", **params) -> dict | None:
        return self._get_url(f"{config.API_BASE}/{path.lstrip('/')}",
                             cacheable=cacheable, cache_tag=cache_tag, **params)

    # ----------------------------------------------------------- team stats
    def team_season_lines(self, season: int
                          ) -> tuple[dict[int, TeamSeasonLine], float | None]:
        """Season-to-date team lines (live use) + league wOBA computed from
        league-wide counting stats. For backtests use team_lines_as_of()."""
        out: dict[int, TeamSeasonLine] = {}
        league_woba = None
        hit = self._get("teams/stats", sportIds=config.SPORT_ID, season=season,
                        group="hitting", stats="season")
        pit = self._get("teams/stats", sportIds=config.SPORT_ID, season=season,
                        group="pitching", stats="season")
        if not hit or not pit:
            log.error("Could not fetch team stats for %s", season)
            return out, None
        try:
            agg = {"uBB": 0, "HBP": 0, "1B": 0, "2B": 0, "3B": 0, "HR": 0,
                   "PA_DEN": 0}
            for sp in hit["stats"][0]["splits"]:
                tid = sp["team"]["id"]
                st = sp["stat"]
                out[tid] = TeamSeasonLine(
                    team_id=tid, name=sp["team"]["name"],
                    games=int(st["gamesPlayed"]),
                    runs_scored=float(st["runs"]),
                    runs_allowed=0.0, innings_pitched=0.0,
                )
                # accumulate league wOBA inputs (for platoon adjustments)
                try:
                    ubb = int(st["baseOnBalls"]) - int(st.get("intentionalWalks", 0))
                    singles = (int(st["hits"]) - int(st["doubles"])
                               - int(st["triples"]) - int(st["homeRuns"]))
                    agg["uBB"] += ubb
                    agg["HBP"] += int(st.get("hitByPitch", 0))
                    agg["1B"] += singles
                    agg["2B"] += int(st["doubles"])
                    agg["3B"] += int(st["triples"])
                    agg["HR"] += int(st["homeRuns"])
                    agg["PA_DEN"] += (int(st["atBats"]) + ubb
                                      + int(st.get("hitByPitch", 0))
                                      + int(st.get("sacFlies", 0)))
                except (KeyError, ValueError):
                    pass
            if agg["PA_DEN"] > 0:
                w = config.WOBA_WEIGHTS
                league_woba = (w["uBB"] * agg["uBB"] + w["HBP"] * agg["HBP"]
                               + w["1B"] * agg["1B"] + w["2B"] * agg["2B"]
                               + w["3B"] * agg["3B"] + w["HR"] * agg["HR"]
                               ) / agg["PA_DEN"]
            for sp in pit["stats"][0]["splits"]:
                tid = sp["team"]["id"]
                st = sp["stat"]
                if tid not in out:
                    continue
                # In the pitching group, 'runs' IS runs allowed (verified:
                # there is no 'runsAllowed' key on the live API).
                out[tid].runs_allowed = float(st["runs"])
                out[tid].innings_pitched = parse_ip(st["inningsPitched"])
        except (KeyError, IndexError, ValueError) as e:
            log.error("Unexpected team stats shape: %s", e)
        log.info("Loaded season lines for %d teams (league wOBA %.3f)",
                 len(out), league_woba or -1)
        return out, league_woba

    # ------------------------------------------------------------- schedule
    def schedule(self, start: date, end: date | None = None,
                 with_lineups: bool = False) -> list[GameInfo]:
        end = end or start
        hydrate = "probablePitcher" + (",lineups" if with_lineups else "")
        data = self._get(
            "schedule", cacheable=end < date.today(),
            sportId=config.SPORT_ID,
            startDate=start.isoformat(), endDate=end.isoformat(),
            hydrate=hydrate,
        )
        games: list[GameInfo] = []
        if not data or not data.get("dates"):
            log.warning("No games scheduled %s..%s", start, end)
            return games
        for d in data["dates"]:
            for g in d["games"]:
                ht, at = g["teams"]["home"], g["teams"]["away"]
                hp, ap = ht.get("probablePitcher"), at.get("probablePitcher")
                lu = g.get("lineups") or {}
                games.append(GameInfo(
                    game_pk=g["gamePk"], game_date=d["date"],
                    game_time_utc=g.get("gameDate", ""),
                    home_id=ht["team"]["id"], away_id=at["team"]["id"],
                    home_name=ht["team"]["name"], away_name=at["team"]["name"],
                    home_prob_id=hp["id"] if hp else None,
                    away_prob_id=ap["id"] if ap else None,
                    home_prob_name=hp["fullName"] if hp else "TBD",
                    away_prob_name=ap["fullName"] if ap else "TBD",
                    state=g["status"]["detailedState"],
                    home_score=ht.get("score"), away_score=at.get("score"),
                    home_lineup=[p["id"] for p in lu.get("homePlayers", [])],
                    away_lineup=[p["id"] for p in lu.get("awayPlayers", [])],
                ))
        return games

    # ----------------------------------------------------- pitcher, no leak
    def pitcher_line(self, pid: int, season: int,
                     as_of: date | None = None) -> PitcherLine | None:
        """Pitcher aggregate line. as_of=D -> ONLY games strictly before D
        (leakage-free backtesting via byDateRange)."""
        if as_of is None:
            params = dict(stats="season", group="pitching", season=season)
            cacheable = False
        else:
            params = dict(stats="byDateRange", group="pitching", season=season,
                          startDate=f"{season}-03-01",
                          endDate=(as_of - timedelta(days=1)).isoformat())
            cacheable = as_of <= date.today()
        data = self._get(f"people/{pid}/stats", cacheable=cacheable, **params)
        try:
            split = data["stats"][0]["splits"][0]
            st = split["stat"]
            name = split.get("player", {}).get("fullName", str(pid))
            return PitcherLine(pitcher_id=pid, name=name,
                               ip=parse_ip(st["inningsPitched"]),
                               runs=float(st.get("runs", 0)))
        except (TypeError, KeyError, IndexError):
            log.warning("No pitching data for pitcher %s (as_of=%s)", pid, as_of)
            return None

    def pitcher_hands(self, pids: list[int]) -> dict[int, str]:
        """Throwing hand for a batch of pitchers (one API call)."""
        pids = [p for p in pids if p]
        if not pids:
            return {}
        data = self._get("people", cacheable=True,
                         personIds=",".join(map(str, pids)))
        out = {}
        for p in (data or {}).get("people", []):
            out[p["id"]] = p.get("pitchHand", {}).get("code", "R")
        missing = set(pids) - set(out)
        if missing:
            log.warning("No handedness for pitchers %s; assuming R", missing)
        return out

    # -------------------------------------------------------- batter splits
    def batter_splits(self, pids: list[int], season: int,
                      cache_tag: str = "") -> dict[int, BatterSplits]:
        """Platoon counting stats vs LHP/RHP for a whole lineup in ONE
        batched call. cache_tag should encode the as-of date, because
        season-to-date splits change daily.

        NOTE (leakage): the API can't give splits 'as of' a past date, so
        backtests using this see slightly-future split data. Team/starter
        cores stay leakage-free; lineup adjustment is optional in backtests
        for exactly this reason."""
        pids = [p for p in pids if p]
        if not pids:
            return {}
        hyd = (f"stats(group=[hitting],type=[statSplits],"
               f"sitCodes=[vl,vr],season={season})")
        data = self._get("people", cacheable=bool(cache_tag),
                         cache_tag=cache_tag,
                         personIds=",".join(map(str, pids)), hydrate=hyd)
        out: dict[int, BatterSplits] = {}
        for p in (data or {}).get("people", []):
            bs = BatterSplits(player_id=p["id"], name=p.get("fullName", "?"),
                              bat_side=p.get("batSide", {}).get("code", "R"))
            for grp in p.get("stats", []):
                for sp in grp.get("splits", []):
                    code = sp.get("split", {}).get("code")
                    if code in ("vl", "vr"):
                        bs.vs[code] = sp.get("stat", {})
            out[p["id"]] = bs
        got = len(out)
        if got < len(pids):
            log.warning("Splits returned for %d/%d batters", got, len(pids))
        return out

    # ------------------------------------------------------- bullpen usage
    def bullpen_usage(self, team_id: int, as_of: date,
                      lookback_days: int = 3) -> BullpenUsage:
        """Relief workload over the trailing `lookback_days` ending the day
        BEFORE as_of (leakage-free). Uses boxscores; the first pitcher in
        the appearance-ordered 'pitchers' list is the starter - excluded."""
        start = as_of - timedelta(days=lookback_days)
        end = as_of - timedelta(days=1)
        usage = BullpenUsage(team_id, 0.0, 0, 0)
        for g in self.schedule(start, end):
            if g.state != "Final" or team_id not in (g.home_id, g.away_id):
                continue
            side = "home" if g.home_id == team_id else "away"
            box = self._get(f"game/{g.game_pk}/boxscore", cacheable=True)
            if not box:
                continue
            try:
                tm = box["teams"][side]
                pitcher_ids = tm.get("pitchers", [])
                for pid in pitcher_ids[1:]:   # skip the starter
                    st = (tm["players"].get(f"ID{pid}", {})
                          .get("stats", {}).get("pitching", {}))
                    usage.relief_ip_3d += parse_ip(st.get("inningsPitched", "0"))
                    usage.relief_pitches_3d += int(st.get("numberOfPitches", 0)
                                                   or st.get("pitchesThrown", 0))
                usage.games_counted += 1
            except (KeyError, TypeError) as e:
                log.warning("Boxscore parse failed for game %s: %s", g.game_pk, e)
        return usage

    # -------------------------------------------------------------- weather
    def weather(self, home_team_id: int, game_time_utc: str) -> Weather | None:
        """Hourly weather at first pitch. Future/today -> forecast API;
        past -> archive API (so backtests use the weather that actually
        happened - the honest stand-in for a bet-time forecast).
        Returns None for domes or on any failure (engine treats as neutral)."""
        venue = config.VENUES.get(home_team_id)
        if venue is None:
            log.warning("No venue info for team %s; skipping weather",
                        home_team_id)
            return None
        lat, lon, roof, _ = venue
        if roof == "dome":
            return None
        try:
            dt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
        except ValueError:
            log.warning("Bad game time %r; skipping weather", game_time_utc)
            return None
        day = dt.date()
        common = dict(latitude=lat, longitude=lon,
                      hourly="temperature_2m,wind_speed_10m,wind_direction_10m",
                      temperature_unit="fahrenheit", wind_speed_unit="mph",
                      timezone="UTC")
        if day >= date.today():
            url, src = "https://api.open-meteo.com/v1/forecast", "forecast"
            data = self._get_url(url, cacheable=False, **common,
                                 start_date=day.isoformat(),
                                 end_date=day.isoformat())
        else:
            url, src = "https://archive-api.open-meteo.com/v1/archive", "archive"
            data = self._get_url(url, cacheable=True, **common,
                                 start_date=day.isoformat(),
                                 end_date=day.isoformat())
        try:
            hours = data["hourly"]["time"]
            target = dt.strftime("%Y-%m-%dT%H:00")
            idx = hours.index(target) if target in hours else min(
                range(len(hours)), key=lambda i: abs(
                    datetime.fromisoformat(hours[i]).replace(tzinfo=timezone.utc)
                    - dt))
            t = data["hourly"]["temperature_2m"][idx]
            ws = data["hourly"]["wind_speed_10m"][idx]
            wd = data["hourly"]["wind_direction_10m"][idx]
            if t is None or ws is None:
                raise ValueError("null weather values")
            return Weather(temp_f=float(t), wind_mph=float(ws),
                           wind_dir_deg=float(wd or 0), source=src)
        except (TypeError, KeyError, ValueError, IndexError) as e:
            log.warning("Weather unavailable for team %s at %s (%s)",
                        home_team_id, game_time_utc, e)
            return None

    # ------------------------------------------------------- umpire lookup
    def hp_umpire(self, game_pk: int, game_final: bool = False) -> str | None:
        """Home-plate umpire from boxscore officials. MLB publishes crews
        shortly before first pitch, so pre-game this usually returns None -
        callers must treat that as neutral, never as an error."""
        box = self._get(f"game/{game_pk}/boxscore", cacheable=game_final)
        for o in (box or {}).get("officials", []):
            if o.get("officialType") == "Home Plate":
                return o.get("official", {}).get("fullName")
        return None

    # ------------------------------------------- team rates for backtesting
    def team_lines_as_of(self, season: int, as_of: date,
                         season_start: date) -> dict[int, TeamSeasonLine]:
        """Rebuild team lines from final scores of games BEFORE as_of
        (leakage-free). IP approximated as 9/game - cancels in the
        ratio-to-league the engine uses."""
        finals = [g for g in self.schedule(season_start, as_of - timedelta(days=1))
                  if g.state == "Final"
                  and g.home_score is not None and g.away_score is not None]
        acc: dict[int, TeamSeasonLine] = {}

        def bump(tid, name, rs, ra):
            t = acc.setdefault(tid, TeamSeasonLine(tid, name, 0, 0.0, 0.0, 0.0))
            t.games += 1
            t.runs_scored += rs
            t.runs_allowed += ra
            t.innings_pitched += 9.0

        for g in finals:
            bump(g.home_id, g.home_name, g.home_score, g.away_score)
            bump(g.away_id, g.away_name, g.away_score, g.home_score)
        log.info("As-of %s: rebuilt rates for %d teams from %d finals",
                 as_of, len(acc), len(finals))
        return acc


# ===========================================================================
# File-based inputs (projections, odds)
# ===========================================================================
def read_csv_rows(path: str | Path) -> tuple[list[dict], list[str]]:
    """Encoding-tolerant csv read -> (rows, fieldnames).

    Player names have accents and csvs come from everywhere (Excel, web
    exports, cross-OS copies), so try utf-8 then cp1252 then latin-1.
    latin-1 maps every byte, so this can't raise on encoding."""
    path = Path(path)
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fields = list(reader.fieldnames or [])
            if enc != "utf-8-sig":
                log.info("%s read with %s fallback encoding", path, enc)
            return rows, fields
        except UnicodeDecodeError:
            continue
        except OSError as e:
            log.error("Cannot read %s: %s", path, e)
            return [], []
    return [], []  # unreachable given latin-1, kept for safety


def load_projections(path: str | Path = config.PROJECTIONS_CSV
                     ) -> dict[int, float]:
    """FanGraphs-style projections CSV -> {mlbam_id: projected RA9 prior}.

    Auto-detects the MLBAM id column (MLBAMID/xMLBAMID/mlbam_id) and a rate
    column (prefers FIP, falls back to ERA, then RA9). Missing file is fine:
    callers fall back to league-average priors (V1 behavior)."""
    path = Path(path)
    if not path.exists():
        log.warning("Projections file %s not found - using league-average "
                    "starter priors (weaker; see config.PROJECTIONS_CSV)", path)
        return {}
    rows, fields = read_csv_rows(path)
    id_cols = ("mlbamid", "xmlbamid", "mlbam_id", "mlb_id", "key_mlbam")
    cols = {c.lower().strip('"'): c for c in fields}
    id_col = next((cols[c] for c in id_cols if c in cols), None)
    rate_col, mult = None, 1.0
    if "fip" in cols:
        rate_col, mult = cols["fip"], config.RA9_FROM_FIP_MULT
    elif "era" in cols:
        rate_col, mult = cols["era"], config.RA9_FROM_FIP_MULT
    elif "ra9" in cols:
        rate_col, mult = cols["ra9"], 1.0
    if not id_col or not rate_col:
        log.error("Projections CSV %s: need an MLBAM id column and a "
                  "FIP/ERA/RA9 column; found %s", path, list(cols))
        return {}
    out: dict[int, float] = {}
    for row in rows:
        try:
            out[int(float(row[id_col]))] = float(row[rate_col]) * mult
        except (ValueError, KeyError, TypeError):
            continue
    log.info("Loaded %d pitcher projections from %s (col=%s)",
             len(out), path, rate_col)
    return out


# ===========================================================================
# Live odds: The Odds API (free tier)
# ===========================================================================
def fetch_live_odds(api_key: str | None = None,
                    csv_path: str | Path = config.ODDS_CSV,
                    record_close: bool = False) -> int:
    """Pull current MLB moneylines from The Odds API into the odds.csv ledger.

    Consensus = median implied prob across books (resists one stale line),
    de-vigged proportionally. Free tier has no line history so the ledger
    builds its own: first sight -> *_open, every run -> current, run with
    record_close=True -> *_close (do that near first pitch or no CLV).
    Returns games updated; 0 + log on any failure."""
    import os
    import statistics

    # Key resolution: explicit arg > environment > local key file.
    api_key = api_key or os.environ.get(config.ODDS_API_KEY_ENV)
    if not api_key:
        key_file = Path(config.ODDS_API_KEY_FILE)
        if key_file.exists():
            api_key = key_file.read_text(encoding="utf-8",
                                         errors="ignore").strip().splitlines()[0].strip()
    if not api_key:
        log.warning("No Odds API key (arg, env %s, or %s) - skipping live odds",
                    config.ODDS_API_KEY_ENV, config.ODDS_API_KEY_FILE)
        return 0
    try:
        r = requests.get(
            f"{config.ODDS_API_BASE}/sports/{config.ODDS_SPORT}/odds",
            params={"apiKey": api_key, "regions": config.ODDS_REGIONS,
                    "markets": "h2h", "oddsFormat": "american"},
            timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
        games = r.json()
        remaining = r.headers.get("x-requests-remaining")
        log.info("Odds API: %d events (%s requests remaining this month)",
                 len(games), remaining)
    except (requests.RequestException, ValueError) as e:
        log.error("Odds API fetch failed: %s", e)
        return 0

    def implied(ml: float) -> float:
        return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)

    def to_american(p: float) -> float:
        p = min(max(p, 0.02), 0.98)
        return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)

    # load existing ledger
    csv_path = Path(csv_path)
    ledger: dict[tuple, dict] = {}
    fields = ["date", "home_team", "away_team", "home_ml", "away_ml",
              "home_ml_open", "away_ml_open", "home_ml_close",
              "away_ml_close", "last_updated"]
    if csv_path.exists():
        for row in read_csv_rows(csv_path)[0]:
            ledger[(row["date"], row["home_team"], row["away_team"])] = row

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    updated = 0
    for g in games:
        home = config.ODDS_TEAM_ALIASES.get(g["home_team"], g["home_team"])
        away = config.ODDS_TEAM_ALIASES.get(g["away_team"], g["away_team"])
        # Odds API commence_time is UTC; MLB schedule dates are US-local.
        # UTC-4 (ET in season) puts late games on the right calendar day.
        ct = datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
        # CRITICAL: once a game starts, the API serves LIVE in-play odds
        # under the same event. Ingesting those would overwrite pre-game
        # lines (and a mistimed --close would record in-game prices as
        # "closing lines", corrupting CLV). Pre-game markets only.
        if ct <= datetime.now(timezone.utc):
            continue
        gdate = (ct - timedelta(hours=4)).date().isoformat()

        h_probs, a_probs = [], []
        for bk in g.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                if g["home_team"] in prices and g["away_team"] in prices:
                    h_probs.append(implied(prices[g["home_team"]]))
                    a_probs.append(implied(prices[g["away_team"]]))
        if not h_probs:
            continue
        mh, ma = statistics.median(h_probs), statistics.median(a_probs)
        s = mh + ma                       # remove consensus vig
        ml_h, ml_a = to_american(mh / s), to_american(ma / s)

        key = (gdate, home, away)
        row = ledger.get(key, {k: "" for k in fields})
        row.update({"date": gdate, "home_team": home, "away_team": away,
                    "home_ml": ml_h, "away_ml": ml_a, "last_updated": now})
        if not row.get("home_ml_open"):
            row["home_ml_open"], row["away_ml_open"] = ml_h, ml_a
        if record_close:
            row["home_ml_close"], row["away_ml_close"] = ml_h, ml_a
        ledger[key] = row
        updated += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ledger.values())
    log.info("Odds ledger: %d games updated (close=%s) -> %s",
             updated, record_close, csv_path)
    return updated


def load_odds_csv(path: str | Path) -> dict[tuple[str, str, str], dict]:
    """Odds CSV keyed by (date, home_team, away_team).

    Required: date,home_team,away_team,home_ml,away_ml   (American odds,
    the line available when you would bet).
    Optional: home_ml_close,away_ml_close                 (for CLV).
    Team names must match MLB API names ('New York Yankees')."""
    book: dict[tuple[str, str, str], dict] = {}
    path = Path(path)
    if not path.exists():
        log.error("Odds file %s not found", path)
        return book
    for r in read_csv_rows(path)[0]:
        try:
            book[(r["date"].strip(), r["home_team"].strip(),
                  r["away_team"].strip())] = r
        except (KeyError, AttributeError) as e:
            log.error("Odds CSV missing column: %s", e)
            return {}
    log.info("Loaded odds for %d games from %s", len(book), path)
    return book
