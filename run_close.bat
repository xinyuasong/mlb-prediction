@echo off
rem MLB model - closing-line snapshot (scheduled task wrapper)
rem Records current lines as CLOSING lines for games that have not started
rem yet. Safe to run multiple times per day: started games keep their last
rem pre-game snapshot. The 45s wait lets Wi-Fi reconnect after a wake-from-
rem sleep so the fetch doesn't fire before DNS is up.
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
timeout /t 45 /nobreak >nul
echo ================ %date% %time% CLOSE SNAPSHOT ================ >> scheduler.log
python daily_runner.py --close >> scheduler.log 2>&1

