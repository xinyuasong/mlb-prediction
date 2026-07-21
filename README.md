# mlb_prediction

MLB game model I've been building. It pulls stats from the MLB API, projects
expected runs for each team, runs 20k Monte Carlo sims per game, and spits out
win probabilities and totals. Then it compares those numbers against actual
sportsbook lines to see where it disagrees with the market, and tracks whether
those disagreements were ever right (spoiler: the market is hard to beat).

Not betting advice. This is mostly a calibration project — the interesting
question is "when the model says 60%, does it happen 60% of the time," not
"can I get rich."

## How it works (short version)

Expected runs for each side = league average × team offense × opposing pitching
× park × weather × umpire × home field. Every input gets regressed toward a
prior before it's trusted — team rates toward league average, starters toward
their ZiPS projection — because small-sample baseball stats lie constantly.
Then each game gets simulated 20,000 times with a negative binomial run
distribution (real scores are streakier than Poisson), proper walk-off logic,
and extra innings with the ghost runner.

Extra stuff layered on top:

- daily lineups with platoon (L/R) splits, heavily regressed
- bullpen fatigue from the last 3 days of boxscores
- weather (temp + wind direction) at outdoor parks
- home plate umpire tendencies (small, capped at ±3%)
- a final calibration pass fitted from backtest results
- an optional LightGBM residual layer that only activates if it beats the
  base model out-of-sample (so far it never has, which is honest of it)

## Does it work?

Backtested on 1,384 games of the 2026 season (walk-forward, no data leakage —
each day only uses stats that existed before that day):

- Brier score 0.2451 (coin flip = 0.2500, always-pick-home = 0.2497)
- log loss 0.6833 (coin = 0.6931)
- predicted home win rate 51.2% vs actual 51.6%
- run totals off by 3.56 per game on average

Translation: better than dumb baselines, picks winners ~56-57%, probably still
a bit behind the closing line. The backtest also caught two real bugs (totals
ran hot from double-counting summer heat, and probabilities were squeezed too
close to 50%) — both fixed and the fixes are documented in the code.

Whether the model's disagreements with sportsbooks mean anything gets settled
by closing line value tracking, which logs automatically every day. Verdict
pending.

## Setup

Python 3.10+.

```
pip install -r requirements.txt
python run_model.py
```

That's enough for predictions. Optional:

- **odds / EV comparison**: free key from the-odds-api.com, drop it in
  `odds_api_key.txt` (or env `ODDS_API_KEY`)
- **discord/slack pings**: set env `MLB_WEBHOOK_URL` or use `--webhook-url`
- **projections**: FanGraphs paywalled their CSV export, so copy/paste the
  projections page into `zips_paste.txt` and run
  `python fetch_projections.py --paste zips_paste.txt`. Worth doing — starter
  priors are the single most valuable input.

## Commands

```
python run_model.py                          # today's slate
python run_model.py --date 2026-07-22        # any date
python run_model.py --odds odds.csv          # + market comparison / EV flags
python run_model.py --verbose                # see every data decision
python run_model.py --no-lineups --no-weather --no-fatigue   # bare-bones mode

python daily_runner.py                       # full pipeline: odds -> slate -> webhook
python daily_runner.py --close               # snapshot closing lines (run before first pitch)

python backtest.py --start 2026-03-25 --end 2026-12-31        # season to date
python backtest.py --start ... --end ... --odds odds.csv      # + EV/CLV eval
python backtest.py --start ... --end ... --ml                 # train/eval the ML layer

python -m mlb_model.ml_layer --train backtest_results/predictions.csv
```

`run_model.py` is read-only, run it whenever. `daily_runner.py` writes to the
odds ledger and prediction history, so that one's meant for the schedule.

On Windows, `setup_schedule.ps1` registers scheduled tasks for all of it:
morning slate at 10:00, closing-line snapshots at 12:45 and 18:40 (they have
to happen *before* first pitch — the pregame market disappears once games
start), weekly backtest on Mondays.

```
powershell -ExecutionPolicy Bypass -File .\setup_schedule.ps1
```

## Files

```
mlb_model/            the package
  config.py           every tunable constant, with notes on why
  data.py             all API/file IO, cached + leakage-safe
  features.py         regression, park adjustment, platoon, fatigue, weather
  engine.py           expected runs
  simulate.py         monte carlo
  ml_layer.py         gated residual model
  report.py           output tables, de-vig, EV
run_model.py          predict a slate
daily_runner.py       the automated version
backtest.py           grade it on history
fetch_projections.py  projections import
odds.csv              line ledger (open/current/close per game)
slate_history.csv     every prediction ever published, machine-readable
backtest_results/     latest backtest output (overwritten each run)
```

Logs go to `mlb_model.log` / `scheduler.log`. `.api_cache/` makes repeat
backtests fast and is safe to delete.

## Things it doesn't know

Injuries and roster news after the morning run, pitch count plans, travel
fatigue, framing, which specific relievers are available, and whatever the
market figures out after lineups drop. Park factors and umpire numbers are
hand-maintained constants. If the model and the market disagree, the smart
default is that the market knows something.
