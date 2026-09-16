@echo off
cd /d "%~dp0"
timeout /t 60 /nobreak >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_results.ps1"
