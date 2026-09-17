@echo off
rem MLB model - morning slate (scheduled task wrapper)
rem The 60s wait matters: when the task wakes the laptop from sleep, Wi-Fi
rem takes a moment to reconnect - on 2026-07-21 the odds fetch fired before
rem DNS was up and the whole slate ran without market data. Cheap insurance.
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
timeout /t 60 /nobreak >nul
echo ================ %date% %time% MORNING SLATE ================ >> scheduler.log
python daily_runner.py >> scheduler.log 2>&1

