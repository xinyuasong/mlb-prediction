"""Walk-forward backtest + market/CLV evaluation.

    python backtest.py --start 2026-03-25 --end 2026-12-31   # clamps to yesterday
    python backtest.py ... --odds odds.csv                   # EV + ROI + CLV
    python backtest.py ... --ml                              # train/eval residual layer

No leakage: team rates for date D come only from games final before D,
starter stats from byDateRange ending D-1, fatigue from prior boxscores,
weather from the historical archive. Exception: platoon splits can only be
fetched season-to-date, so --lineups is off by default (mild leakage,
slightly flattering numbers if you turn it on).

CLV = did the closing line move toward our flagged side vs the line at bet
time. That's the real test of edge - it converges way faster than ROI,
which is noise for months. Needs *_close columns in the odds csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from mlb_model import config, setup_logging
from mlb_model.data import MLBDataClient, load_odds_csv, load_projections
from mlb_model.engine import build_game_context, expected_runs
from mlb_model.features import LeagueContext, calibrate_prob
from mlb_model.report import american_implied, devig_proportional, market_compare
from mlb_model.simulate import simulate_game

log = logging.getLogger("mlb_model.backtest")


# --------------------------------------------------------------------- run
def run_backtest(start: date, end: date, season: int, total_line: float,
                 use_lineups: bool, use_weather: bool,
                 use_fatigue: bool) -> list[dict]:
    client = MLBDataClient()
    season_start = date(season, 3, 15)
    projections = load_projections()
    rng = np.random.default_rng(config.RNG_SEED)
    rows: list[dict] = []
    # Early-season skips are EXPECTED, not errors: a leakage-free backtest
    # can only rate a team once it has MIN_TEAM_GAMES (15) of real history,
    # so roughly the first two weeks of the season produce no predictions.
    # We count those quietly and report once at the end instead of spamming
    # one warning per game.
    skipped_days = 0
    skipped_games = 0
    first_pred_date: str | None = None

    d = start
    while d <= end:
        games = [g for g in client.schedule(d, with_lineups=use_lineups)
                 if g.state == "Final" and g.home_score is not None]
        if not games:
            d += timedelta(days=1)
            continue
        try:
            ctx = LeagueContext(client.team_lines_as_of(season, d, season_start),
                                projections=projections)
        except ValueError:
            log.info("Skipping %s: insufficient pre-date history (expected "
                     "early in season)", d)
            skipped_days += 1
            skipped_games += len(games)
            d += timedelta(days=1)
            continue

        for g in games:
            # skip quietly if either team hasn't reached MIN_TEAM_GAMES yet
            if ctx.rating(g.home_id) is None or ctx.rating(g.away_id) is None:
                skipped_games += 1
                continue
            gc = build_game_context(client, ctx, g, season, as_of=d,
                                    use_lineups=use_lineups,
                                    use_weather=use_weather,
                                    use_fatigue=use_fatigue)
            pred = expected_runs(ctx, g, gc)
            if pred is None:
                continue
            if first_pred_date is None:
                first_pred_date = d.isoformat()
            sim = simulate_game(pred.lam_home, pred.lam_away, rng=rng)
            rows.append({
                "date": d.isoformat(),
                "home": g.home_name, "away": g.away_name,
                # structural outputs = ML feature columns (config.ML_FEATURES)
                # p_home is post-calibration: future backtests measure the
                # pipeline as it actually predicts, and a refit of
                # CALIBRATION_LOGIT_SCALE should then come out ~1.0.
                "p_home": calibrate_prob(sim.p_home),
                "p_home_raw": round(sim.p_home, 5),
                "lam_home": round(pred.lam_home, 3),
                "lam_away": round(pred.lam_away, 3),
                "exp_total": round(sim.exp_total, 3),
                "park_factor": pred.park_factor,
                "weather_mult": round(pred.weather_mult, 4),
                "fatigue_home": round(pred.fatigue_home, 2),
                "fatigue_away": round(pred.fatigue_away, 2),
                "hfa_extras": round(sim.p_extra_innings, 4),
                "p_over": sim.p_over(total_line),
                # outcomes
                "home_won": int(g.home_score > g.away_score),
                "actual_total": g.home_score + g.away_score,
            })
        log.warning("Backtested %s: %d finals (running total %d)",
                    d, len(games), len(rows))
        d += timedelta(days=1)

    if skipped_days or skipped_games:
        log.warning(
            "Skipped %d game(s) across %d early day(s): teams need %d games "
            "of pre-date history before the leakage-free backtest will rate "
            "them (by design). First predicted date: %s",
            skipped_games, skipped_days, config.MIN_TEAM_GAMES,
            first_pred_date or "none")
    return rows


# ----------------------------------------------------------------- metrics
def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(p, y, edges=None):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = edges or [0, .35, .40, .45, .50, .55, .60, .65, 1.0]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        n = int(m.sum())
        se = math.sqrt(p[m].mean() * (1 - p[m].mean()) / n)
        out.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": n,
                    "predicted": float(p[m].mean()),
                    "realized": float(y[m].mean()), "noise_se": se})
    return out


def evaluate(rows: list[dict], outdir: Path, prob_col: str = "p_home") -> dict:
    p = np.array([r[prob_col] for r in rows], float)
    y = np.array([r["home_won"] for r in rows])
    xt = np.array([r["exp_total"] for r in rows], float)
    at = np.array([r["actual_total"] for r in rows], float)
    home_rate = float(y.mean())
    res = {
        "n_games": len(rows),
        "brier": brier(p, y),
        "brier_coin": brier(np.full_like(p, 0.5), y),
        "brier_home_rate": brier(np.full_like(p, home_rate), y),
        "log_loss": log_loss(p, y),
        "log_loss_coin": log_loss(np.full_like(p, 0.5), y),
        "realized_home_rate": home_rate,
        "mean_predicted_home": float(p.mean()),
        "totals_mae": float(np.abs(xt - at).mean()),
        "totals_bias": float((xt - at).mean()),
        "calibration": calibration_table(p, y),
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cal = res["calibration"]
        fig, ax = plt.subplots(figsize=(6, 6))
        xs = [c["predicted"] for c in cal]
        ys = [c["realized"] for c in cal]
        ns = [c["n"] for c in cal]
        ax.plot([0.2, 0.8], [0.2, 0.8], "k--", lw=1, label="perfect")
        ax.scatter(xs, ys, s=[max(20, n) for n in ns], zorder=3)
        for x_, y_, n_ in zip(xs, ys, ns):
            ax.errorbar(x_, y_, yerr=1.96 * math.sqrt(max(y_ * (1 - y_), .04) / n_),
                        fmt="none", ecolor="gray", capsize=3)
            ax.annotate(f"n={n_}", (x_, y_), textcoords="offset points",
                        xytext=(6, -10), fontsize=8)
        ax.set_xlabel("Predicted home win probability")
        ax.set_ylabel("Realized frequency")
        ax.set_title(f"Calibration ({len(rows)} games), 95% binomial noise bars")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "calibration.png", dpi=140)
        res["calibration_plot"] = str(outdir / "calibration.png")
    except ImportError:
        log.warning("matplotlib not installed - skipping calibration plot")
    return res


# ---------------------------------------------------------- market: EV/CLV
def market_eval(rows: list[dict], odds_csv: Path,
                prob_col: str = "p_home") -> dict | None:
    """EV betting simulation + CLV tracking for flagged spots."""
    book = load_odds_csv(odds_csv)
    if not book:
        return None

    def payout(ml):
        ml = float(ml)
        return ml / 100 if ml > 0 else 100 / -ml

    matched = 0
    bets = []          # every HIGH EV flag: (won, ml, clv or None)
    for r in rows:
        o = book.get((r["date"], r["home"], r["away"]))
        if not o:
            continue
        matched += 1
        try:
            mh, ma = devig_proportional(o["home_ml"], o["away_ml"])
        except (ValueError, TypeError):
            continue
        p = float(r[prob_col])
        for side, p_side, p_mkt, ml_key in (("home", p, mh, "home_ml"),
                                            ("away", 1 - p, ma, "away_ml")):
            edge = p_side - p_mkt
            ev = p_side * payout(o[ml_key]) - (1 - p_side)
            if edge >= config.EV_THRESHOLD and ev > 0:
                won = (r["home_won"] == 1) == (side == "home")
                clv = None
                if o.get("home_ml_close") and o.get("away_ml_close"):
                    ch, ca = devig_proportional(o["home_ml_close"],
                                                o["away_ml_close"])
                    p_close = ch if side == "home" else ca
                    p_bet = p_mkt
                    clv = p_close - p_bet   # + = close moved toward us
                bets.append({"date": r["date"], "side": side,
                             "matchup": f"{r['away']} @ {r['home']}",
                             "edge": edge, "ml": float(o[ml_key]),
                             "won": won, "clv": clv})
    if matched == 0:
        log.warning("No odds rows matched backtest games - check team names "
                    "match MLB API names exactly")
        return None

    n = len(bets)
    pnl = sum(payout(b["ml"]) if b["won"] else -1.0 for b in bets)
    clvs = [b["clv"] for b in bets if b["clv"] is not None]
    res = {
        "games_with_odds": matched,
        "high_ev_bets": n,
        "flat_stake_pnl_units": round(pnl, 2),
        "roi": round(pnl / n, 4) if n else None,
        "roi_noise_se_approx": round(1 / math.sqrt(n), 4) if n else None,
    }
    if clvs:
        arr = np.array(clvs)
        res["clv"] = {
            "n_with_close": len(clvs),
            "avg_clv_prob_pts": round(float(arr.mean()), 4),
            "pct_beating_close": round(float((arr > 0).mean()), 3),
            "clv_se": round(float(arr.std(ddof=1) / math.sqrt(len(arr))), 4)
            if len(arr) > 1 else None,
            "interpretation": (
                "positive avg CLV = our flags anticipate line movement = "
                "real information. Avg CLV within ~2 SE of zero = flags are "
                "noise regardless of ROI."),
        }
    else:
        res["clv"] = ("no *_close columns in odds CSV - add home_ml_close/"
                      "away_ml_close to enable CLV, the metric that actually "
                      "matters")
    res["note"] = ("ROI over short windows is variance. CLV is the test: "
                   "a model that flags EV but loses CLV is fooling itself.")
    return res


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--total-line", type=float, default=8.5)
    ap.add_argument("--odds", type=Path, default=None)
    ap.add_argument("--ml", action="store_true",
                    help="train + walk-forward-evaluate the residual layer "
                         "on this window's predictions")
    ap.add_argument("--lineups", action="store_true",
                    help="enable lineup adjustment (mild split leakage - "
                         "see module docstring)")
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--no-fatigue", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("backtest_results"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    season = args.season or args.start.year

    # Clamp the window to completed days: future dates have no finals, and
    # today's games may still be in progress. Lets schedulers pass a far
    # future --end ("season to date") without locale-fragile date math.
    yesterday = date.today() - timedelta(days=1)
    if args.end > yesterday:
        args.end = yesterday

    setup_logging(args.verbose, logfile="backtest.log")
    args.out.mkdir(exist_ok=True)
    if args.lineups:
        log.warning("Lineup adjustment ON in backtest: platoon splits are "
                    "season-to-date (mild leakage) - results will flatter")

    rows = run_backtest(args.start, args.end, season, args.total_line,
                        use_lineups=args.lineups,
                        use_weather=not args.no_weather,
                        use_fatigue=not args.no_fatigue)
    if not rows:
        raise SystemExit("No games backtested - check the date range.")

    with open(args.out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    res = evaluate(rows, args.out)

    # ----- V3: residual ML layer, walk-forward, ship-gated
    if args.ml:
        from mlb_model.ml_layer import ResidualModel
        m = ResidualModel()
        report = m.fit(rows)
        res["ml_walk_forward"] = {
            "structural_log_loss": report.structural_log_loss,
            "ml_log_loss": report.ml_log_loss,
            "shipped": report.shipped, "folds": report.fold_details,
        }
        m.save(args.out / "ml_residual_model.pkl")
        print("\n" + report.summary())

    # ----- V3: market EV + CLV
    if args.odds:
        mkt = market_eval(rows, args.odds)
        if mkt:
            res["market_eval"] = mkt

    (args.out / "metrics.json").write_text(json.dumps(res, indent=2))

    # ----- console report
    print(f"\n=== Backtest {args.start}..{args.end}  ({res['n_games']} games) ===")
    print(f"Brier score : {res['brier']:.4f}   (coin {res['brier_coin']:.4f}, "
          f"home-rate {res['brier_home_rate']:.4f})")
    print(f"Log loss    : {res['log_loss']:.4f}   (coin {res['log_loss_coin']:.4f})")
    print(f"Home teams  : predicted {res['mean_predicted_home']:.1%} vs "
          f"realized {res['realized_home_rate']:.1%}")
    print(f"Totals      : MAE {res['totals_mae']:.2f}, "
          f"bias {res['totals_bias']:+.2f} (+ = model too high)")
    print("\nCalibration (predicted vs realized):")
    for c in res["calibration"]:
        gap = c["realized"] - c["predicted"]
        flag = "" if abs(gap) < 2 * c["noise_se"] else "  <-- outside noise"
        print(f"  {c['bin']:>10s} {c['n']:>5d} {c['predicted']:>6.1%} "
              f"{c['realized']:>6.1%}{flag}")
    if "market_eval" in res:
        print("\nMarket evaluation:")
        print(json.dumps(res["market_eval"], indent=2))
    print(f"\nArtifacts in {args.out}/ - predictions.csv doubles as ML "
          "training data.")
    print("Reminder: per-bin calibration noise is large at small n; a month "
          "is a hint, a season is evidence.")


if __name__ == "__main__":
    main()
