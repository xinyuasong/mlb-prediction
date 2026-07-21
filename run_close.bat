@echo off
rem MLB model - closing-line snapshot (scheduled task wrapper)
rem Records current lines as CLOSING lines for games that have not started
rem yet. Safe to run multiple times per day: games already underway keep
rem their last pre-game snapshot (they vanish from the odds feed), so the
rem 12:45 run locks day games and the 18:40 run locks night games.
cd /d "%~dp0"
echo ================ %date% %time% CLOSE SNAPSHOT ================ >> scheduler.log
python daily_runner.py --close >> scheduler.log 2>&1
