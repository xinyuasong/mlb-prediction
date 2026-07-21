@echo off
rem MLB model - weekly evaluation (scheduled task wrapper)
rem Re-runs the season-to-date, leakage-free backtest joined against the
rem accumulated odds ledger: Brier, log loss, calibration curve, and - once
rem enough closing lines exist - the CLV verdict on HIGH-EV flags.
rem backtest.py clamps the end date to yesterday automatically, so the far
rem future --end below simply means "season to date".
rem Mostly served from .api_cache: a few minutes, not 26.
rem Note: overwrites backtest_results\ - copy that folder first if you want
rem to preserve a particular week's snapshot.
cd /d "%~dp0"
echo ================ %date% %time% WEEKLY EVAL ================ >> scheduler.log
python backtest.py --start 2026-03-25 --end 2026-12-31 --odds odds.csv >> scheduler.log 2>&1
