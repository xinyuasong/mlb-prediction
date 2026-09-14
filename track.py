"""Where are today's flags vs the market right now?

Joins the official flags (slate_history.csv) with the odds ledger and shows
line drift since bet-time: toward the model (+) or away (-). Positive drift
on a flag = the market is agreeing with us = realized CLV building.

    python track.py                 # today's flags
    python track.py 2026-07-21      # any recorded date
"""
import csv
import sys
from datetime import date
from pathlib import Path


def implied(ml):
    ml = float(ml)
    return 100/(ml+100) if ml > 0 else -ml/(-ml+100)


def devig(h, a):
    ih, ia = implied(h), implied(a)
    s = ih + ia
    return (ih/s, ia/s) if s > 0 else (0.5, 0.5)


def rows_for(path, want_date):
    if not Path(path).exists():
        return []
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return [r for r in csv.DictReader(f) if r.get("date") == want_date]
        except UnicodeDecodeError:
            continue
    return []


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()

    hist = rows_for("slate_history.csv", d)
    if not hist:
        print(f"no logged predictions for {d} yet - run daily_runner first")
        return
    # keep the latest logged run per matchup (re-runs append)
    latest = {}
    for r in hist:
        latest[(r["away"], r["home"])] = r
    flags = [r for r in latest.values() if r.get("high_ev") == "1"]
    if not flags:
        print(f"{d}: no HIGH-EV flags logged")
        return

    book = {(r["home_team"], r["away_team"]): r for r in rows_for("odds.csv", d)}

    print(f"\n  flag drift for {d}  (+ = market moving toward the model)\n")
    print(f"  {'FLAG':22s} {'bet-time':>8s} {'now':>8s} {'close':>8s} {'drift':>7s}")
    tot, n = 0.0, 0
    for r in flags:
        o = book.get((r["home"], r["away"]))
        side = r["best_side"]
        try:
            p_bet = devig(r["home_ml"], r["away_ml"])[0 if side == "home" else 1]
        except (ValueError, KeyError):
            continue
        name = (r["home"] if side == "home" else r["away"])[:20]
        cur = cls = None
        if o:
            try:
                cur = devig(o["home_ml"], o["away_ml"])[0 if side == "home" else 1]
            except ValueError:
                pass
            if o.get("home_ml_close"):
                try:
                    cls = devig(o["home_ml_close"], o["away_ml_close"])[0 if side == "home" else 1]
                except ValueError:
                    pass
        ref = cls if cls is not None else cur
        drift = (ref - p_bet) if ref is not None else None
        if drift is not None:
            tot += drift
            n += 1
        print(f"  {name:22s} {p_bet:>7.1%} "
              f"{cur if cur is not None else float('nan'):>7.1%} "
              f"{(f'{cls:.1%}' if cls is not None else '   --'):>8s} "
              f"{(f'{drift:+.1%}' if drift is not None else '    ?'):>7s}")
    if n:
        print(f"\n  avg drift: {tot/n:+.2%} across {n} flag(s)")
        print("  (vs close where recorded, else vs current line)")


if __name__ == "__main__":
    main()
