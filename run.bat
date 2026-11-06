@echo off
REM ==========================================================================
REM  AquaSentinel launcher - just double-click this file.
REM  It finds Python, installs the libraries the first time, then starts the
REM  dashboard and opens it in your browser. Close this window to stop the app.
REM ==========================================================================
title AquaSentinel
cd /d "%~dp0"

REM Use the project's virtual environment if there is one, otherwise system Python
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found.
    echo Install Python 3.10+ from https://www.python.org and tick "Add Python to PATH".
    pause
    exit /b 1
)

REM Install the libraries only if they are missing (first run on a new computer)
"%PY%" -c "import streamlit, sklearn, networkx, plotly, pandas, numpy, joblib" >nul 2>&1
if errorlevel 1 (
    echo Installing required libraries - this happens only once...
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Library installation failed. See the messages above.
        pause
        exit /b 1
    )
)

echo.
echo  Starting AquaSentinel - your browser will open at http://localhost:8501
echo  Keep this window open while you use the app. Close it to stop the app.
echo.
"%PY%" app.py
pause
