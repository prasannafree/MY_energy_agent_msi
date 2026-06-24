@echo off
title EnergyPlus MCP Agent
echo ============================================================
echo   EnergyPlus MCP Agent - Starting...
echo ============================================================
echo.

cd /d "%~dp0"

REM Check if venv exists, if not create one
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo ============================================================
echo   Starting agent server on http://localhost:5000
echo   Press Ctrl+C to stop
echo ============================================================
echo.

python agent.py
