@echo off
title AI Model Compiler & Optimization Engine - Dashboard Launcher
cd /d "%~dp0"

echo ===================================================================
echo   AI Model Compiler ^& Optimization Engine - Web Dashboard
echo ===================================================================

where python >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Python found. Launching server.py...
    python server.py
) else (
    echo [INFO] Starting dashboard in default browser (client engine)...
    start "" "%~dp0frontend\index.html"
)
