"""Automated daily pipeline: odds refresh -> model -> markdown -> webhook.

    python daily_runner.py                       # full morning run
    python daily_runner.py --close               # snapshot closing lines (CLV)
    python daily_runner.py --date 2026-07-21 --no-webhook

Env: ODDS_API_KEY, MLB_WEBHOOK_URL. Scheduler-safe: off-days, missing data,
and webhook failures all log and exit 0 instead of crashing the schedule.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import date
from pathlib import Path

import numpy as np
import requests

from mlb_model import config, setup_logging
from mlb_model.data import (MLBDataClient, fetch_live_odds, load_odds_csv,
                            load_projections)
from mlb_model.engine import build_game_context, expected_runs
from mlb_model.features import LeagueContext, calibrate_prob
from mlb_model.report import fair_moneyline, market_compare, team_abbr
from mlb_model.simulate import simulate_game

log = logging.getLogger("mlb_model.daily_runner")


# ----------------------------------------------------------------- pipeline
def run_slate(run_date: date, total_line: float = 8.5,
              odds_csv: str = config.ODDS_CSV) -> list:
    """Full V2 pipeline for one slate. Returns [] on an off-day."""
    season = run_date.year
    client = MLBDataClient()

    lines, league_woba = client.team_season_lines(season)
    if not lines:
        log.error("Team stats unavailable - aborting slate")
        return []
    ctx = LeagueContext(lines, league_woba=league_woba,
                        projections=load_projections())

    games = client.schedule(run_date, with_lineups=True)
    if not games:
        return []

    ml_model = None
    try:
        from mlb_model.ml_layer import ResidualModel
        ml_model = ResidualModel.load()
    except Exception as e:  # noqa: BLE001
        log.warning("ML layer unavailable: %s", e)

    book = load_odds_csv(odds_csv) if Path(odds_csv).exists() else {}
    rng = np.random.default_rng(config.RNG_SEED)
    preds = []
    for g in games:
        try:
            gc = build_game_context(client, ctx, g, season)
            pred = expected_runs(ctx, g, gc)
            if pred is None:
                continue
            sim = simulate_game(pred.lam_home, pred.lam_away, rng=rng)
            pred.p_home = calibrate_prob(sim.p_home)
            pred.exp_total = sim.exp_total
            pred.p_over = sim.p_over(total_line)
            pred.total_line = total_line
            if ml_model is not None:
                pred.p_home_ml = ml_model.predict({
                    "p_home": sim.p_home, "lam_home": pred.lam_home,
                    "lam_away": pred.lam_away, "exp_total": sim.exp_total,
                    "park_factor": pred.park_factor,
                    "weather_mult": pred.weather_mult,
                    "fatigue_home": pred.fatigue_home,
                    "fatigue_away": pred.fatigue_away,
                    "hfa_extras": sim.p_extra_innings,
                })
            row = book.get((g.game_date, g.home_name, g.away_name))
            if row:
                market_compare(pred, row)
            preds.append(pred)
        except Exception as e:  # noqa: BLE001 - one bad game != dead slate
            log.error("Game %s @ %s failed: %s", g.away_name, g.home_name, e)
    return preds


# ------------------------------------------------------------ slate history
def append_slate_history(preds: list, path: str = "slate_history.csv") -> None:
    """Append today's published predictions to a permanent machine-readable
    ledger. scheduler.log keeps the human-readable slate; this file is the
    exact record of what the model said each morning, joinable later against
    results (MLB API) and closing lines (odds.csv) for grading. Append-only
    with a run timestamp: re-runs add rows rather than rewriting history."""
    import csv
    from datetime import datetime, timezone

    fields = ["run_ts", "date", "away", "home", "p_home", "exp_total",
              "p_over", "total_line", "home_ml", "away_ml",
              "market_p_home", "best_side", "best_edge", "high_ev"]
    new_file = not Path(path).exists()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
            for p in preds:
                m = p.market or {}
                w.writerow({
                    "run_ts": ts, "date": p.game.game_date,
                    "away": p.game.away_name, "home": p.game.home_name,
                    "p_home": round(p.p_home, 5),
                    "exp_total": round(p.exp_total, 2),
                    "p_over": round(p.p_over, 4),
                    "total_line": p.total_line,
                    "home_ml": m.get("home_ml", ""),
                    "away_ml": m.get("away_ml", ""),
                    "market_p_home": round(m["market_p_home"], 4) if m else "",
                    "best_side": m.get("best_side", ""),
                    "best_edge": round(m["best_edge"], 4) if m else "",
                    "high_ev": int(m.get("high_ev", False)) if m else "",
                })
        log.info("Appended %d predictions to %s", len(preds), path)
    except OSError as e:
        log.error("Could not write slate history: %s", e)


# ----------------------------------------------------------------- markdown
def markdown_summary(preds: list, run_date: date, total_line: float) -> str:
    if not preds:
        return (f"**MLB Model — {run_date}**\n\nNo games today "
                "(off-day or data unavailable).")
    lines = [f"**MLB Model — {run_date}** ({len(preds)} games)", ""]

    flagged = [p for p in preds if p.market and p.market["high_ev"]]
    if flagged:
        lines.append(f"🎯 **HIGH-EV candidates (edge ≥ "
                     f"{config.EV_THRESHOLD:.1%})**")
        lines.append("")
        lines.append("| Side | ML | Model | Market | Edge | EV/unit |")
        lines.append("|---|---|---|---|---|---|")
        for p in sorted(flagged, key=lambda x: -x.market["best_edge"]):
            m = p.market
            side = (p.game.home_name if m["best_side"] == "home"
                    else p.game.away_name)
            pm = p.p_home_ml if p.p_home_ml is not None else p.p_home
            p_side = pm if m["best_side"] == "home" else 1 - pm
            p_mkt = (m["market_p_home"] if m["best_side"] == "home"
                     else 1 - m["market_p_home"])
            lines.append(f"| **{side}** | {m['best_ml']:+.0f} | {p_side:.1%} "
                         f"| {p_mkt:.1%} | {m['best_edge']:+.1%} "
                         f"| {m['best_ev']:+.3f} |")
        lines.append("")
        lines.append("_Edge = model minus de-vigged market. Only actionable "
                     "if calibration + CLV history support it (backtest.py). "
                     "The market is usually right._")
    else:
        lines.append("_No HIGH-EV spots vs current lines (the normal "
                     "outcome)._" if any(p.market for p in preds) else
                     "_No market odds loaded (set ODDS_API_KEY or provide "
                     "odds.csv)._")
    lines.append("")

    lines.append("| Matchup | H Win | Fair ML (A/H) | xTot | "
                 f"O{total_line} | Notes |")
    lines.append("|---|---|---|---|---|---|")
    for p in preds:
        g = p.game
        pm = p.p_home_ml if p.p_home_ml is not None else p.p_home
        a_sp = g.away_prob_name.split()[-1] if g.away_prob_name != "TBD" else "TBD"
        h_sp = g.home_prob_name.split()[-1] if g.home_prob_name != "TBD" else "TBD"
        matchup = (f"{team_abbr(g.away_id, g.away_name)} {a_sp} @ "
                   f"{team_abbr(g.home_id, g.home_name)} {h_sp}")
        notes = []
        if p.weather_mult != 1.0:
            notes.append(f"wx {(p.weather_mult - 1) * 100:+.0f}%")
        if p.umpire_mult != 1.0:
            notes.append(f"ump {(p.umpire_mult - 1) * 100:+.0f}%")
        if p.fatigue_home > 1:
            notes.append("pen-fat H")
        if p.fatigue_away > 1:
            notes.append("pen-fat A")
        if "no/low" in p.home_starter_note or "no/low" in p.away_starter_note:
            notes.append("SP?")
        if not g.home_lineup:
            notes.append("no lineup")
        lines.append(
            f"| {matchup} "
            f"| {pm:.1%} | {fair_moneyline(1 - pm)}/{fair_moneyline(pm)} "
            f"| {p.exp_total:.1f} | {p.p_over:.0%} "
            f"| {' · '.join(notes) if notes else '—'} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ webhook
def post_webhook(url: str, markdown: str) -> bool:
    """POST to a Discord or Slack incoming webhook. Discord truncates at
    2000 chars, so long slates are chunked on line boundaries. Returns
    success; failures are logged, never raised."""
    is_discord = "discord.com" in url or "discordapp.com" in url
    chunks, cur = [], ""
    for line in markdown.splitlines(keepends=True):
        if len(cur) + len(line) > config.DISCORD_CHAR_LIMIT:
            chunks.append(cur)
            cur = ""
        cur += line
    if cur:
        chunks.append(cur)

    ok = True
    for chunk in chunks:
        payload = {"content": chunk} if is_discord else {"text": chunk}
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code >= 300:
                log.error("Webhook HTTP %s: %s", r.status_code, r.text[:200])
                ok = False
        except requests.RequestException as e:
            log.error("Webhook post failed: %s", e)
            ok = False
    return ok


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, default=date.today())
    ap.add_argument("--total-line", type=float, default=8.5)
    ap.add_argument("--odds-key", default=None,
                    help=f"The Odds API key (or env {config.ODDS_API_KEY_ENV})")
    ap.add_argument("--close", action="store_true",
                    help="record current lines as CLOSING lines, then exit "
                         "(run near first pitch; required for CLV)")
    ap.add_argument("--webhook-url", default=None,
                    help=f"Discord/Slack webhook (or env {config.WEBHOOK_URL_ENV})")
    ap.add_argument("--no-webhook", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose, logfile="daily_runner.log")

    # 1. refresh the odds ledger (no key -> logged skip, model still runs)
    fetch_live_odds(api_key=args.odds_key, record_close=args.close)
    if args.close:
        print("Closing lines recorded in odds.csv - done.")
        return

    # 2. run the slate
    preds = run_slate(args.date, args.total_line)
    if preds:
        append_slate_history(preds)

    # 3. report
    md = markdown_summary(preds, args.date, args.total_line)
    print(md)

    url = args.webhook_url or os.environ.get(config.WEBHOOK_URL_ENV)
    if url and not args.no_webhook:
        sent = post_webhook(url, md)
        print(f"\n[webhook {'sent' if sent else 'FAILED - see log'}]")
    elif not url:
        log.info("No webhook URL configured (set %s) - printed only",
                 config.WEBHOOK_URL_ENV)


if __name__ == "__main__":
    main()
