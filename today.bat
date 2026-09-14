@echo off
rem One-shot day card: refresh odds -> full model -> best bets + slate.
rem Same as the scheduled morning run, minus the webhook. Run any time;
rem later runs use fresher lineups/lines than the 10:00 one.
cd /d "%~dp0"
python daily_runner.py --no-webhook
pause
