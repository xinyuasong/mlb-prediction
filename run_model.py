"""
Daily slate prediction CLI (V2/V3).

    python run_model.py                          # today, all V2 features
    python run_model.py --date 2026-07-20
    python run_model.py --odds odds.csv          # market compare + EV flags
    python run_model.py --ml-model ml_residual_model.pkl
    python run_model.py --no-lineups --no-weather --no-fatigue   # V1 mode
    python run_model.py --projections steamer.csv --verbose
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np

from mlb_model import config, setup_logging
from mlb_model.data import MLBDataClient, load_odds_csv, load_projections
from mlb_model.engine import build_game_context, expected_runs
from mlb_model.features import LeagueContext, calibrate_prob
from mlb_model.report import ev_report, market_compare, slate_footer, slate_table
from mlb_model.simulate import simulate_game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, default=date.today())
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--total-line", type=float, default=8.5)
    ap.add_argument("--projections", default=config.PROJECTIONS_CSV)
    ap.add_argument("--odds", default=None,
                    help="odds CSV -> market comparison + EV flags")
    ap.add_argument("--ml-model", default=config.ML_MODEL_FILE,
                    help="saved residual model (ignored if absent/gated)")
    ap.add_argument("--no-lineups", action="store_true")
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--no-fatigue", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    season = args.season or args.date.year

    log = setup_logging(args.verbose)
    client = MLBDataClient()

    lines, league_woba = client.team_season_lines(season)
    if not lines:
        raise SystemExit("No team data available - aborting.")
    ctx = LeagueContext(lines, league_woba=league_woba,
                        projections=load_projections(args.projections))

    games = client.schedule(args.date, with_lineups=not args.no_lineups)
    if not games:
        raise SystemExit(f"No games on {args.date}.")
    log.info("Slate %s: %d games", args.date, len(games))

    # Optional V3 residual model (silently structural if gated/missing).
    ml_model = None
    try:
        from mlb_model.ml_layer import ResidualModel
        ml_model = ResidualModel.load(args.ml_model)
    except Exception as e:  # noqa: BLE001
        log.warning("ML layer unavailable: %s", e)

    book = load_odds_csv(args.odds) if args.odds else {}

    rng = np.random.default_rng(config.RNG_SEED)
    preds = []
    for g in games:
        gc = build_game_context(client, ctx, g, season,
                                use_lineups=not args.no_lineups,
                                use_weather=not args.no_weather,
                                use_fatigue=not args.no_fatigue)
        pred = expected_runs(ctx, g, gc)
        if pred is None:
            continue
        sim = simulate_game(pred.lam_home, pred.lam_away, rng=rng)
        pred.p_home = calibrate_prob(sim.p_home)
        pred.exp_total = sim.exp_total
        pred.p_over = sim.p_over(args.total_line)
        pred.total_line = args.total_line

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

        odds_row = book.get((g.game_date, g.home_name, g.away_name))
        if odds_row:
            market_compare(pred, odds_row)
        preds.append(pred)

    ml_note = ""
    if ml_model is not None:
        ml_note = (" · ML layer ACTIVE" if ml_model.shipped
                   else " · ML layer gated off (structural)")
    print(f"\n  MLB MODEL SLATE — {args.date}")
    print(f"  league {ctx.league_rs_pg:.2f} R/G · {len(preds)} games"
          f"{ml_note}\n")
    print(slate_table(preds, args.total_line))
    if book:
        print(ev_report(preds))
    print(slate_footer())


if __name__ == "__main__":
    main()
