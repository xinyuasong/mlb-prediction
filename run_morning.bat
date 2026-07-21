@echo off
rem MLB model - morning slate (scheduled task wrapper)
rem Runs the full pipeline: odds refresh -> slate -> webhook (if configured).
rem Output is appended to scheduler.log so silent scheduled runs stay auditable.
cd /d "%~dp0"
echo ================ %date% %time% MORNING SLATE ================ >> scheduler.log
python daily_runner.py >> scheduler.log 2>&1
