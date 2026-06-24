@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  EnergyPlus MCP Agent Setup (Windows)
echo ============================================================

:: 1. Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.9+ and try again.
    pause
    exit /b 1
)

:: 2. Create virtual environment
if not exist venv (
    echo Creating virtual environment (venv)...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment (venv) already exists.
)

:: 3. Install dependencies
echo Installing dependencies from requirements.txt...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo Dependencies installed successfully.

:: 4. Create .env if not exists
if not exist .env (
    echo Creating .env configuration from template...
    copy .env.example .env >nul
    echo.
    echo [IMPORTANT] A new '.env' file has been created.
    echo Please open '.env' and set your GOOGLE_API_KEY.
    echo.
) else (
    echo '.env' configuration file already exists.
)

echo ============================================================
echo  Setup Completed Successfully!
echo ============================================================
echo To start the agent server, run:
echo    start_agent.bat
echo or:
echo    venv\Scripts\python.exe agent.py
echo ============================================================
pause
