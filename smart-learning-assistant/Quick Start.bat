@echo off
title DIP AI Tutor — Quick Start
cd /d "%~dp0"

echo ============================================================
echo   DIP AI Tutor — Quick Start Launcher
echo ============================================================
echo.

:: ── 1. Verify venv exists ────────────────────────────────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\
    echo         Run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: ── 2. Activate venv in this shell (for env variable inheritance) ────────
echo [1/4] Activating virtual environment...
call .venv\Scripts\activate.bat

:: ── 3. Start FastAPI backend in a new terminal window ───────────────────
echo [2/4] Starting FastAPI backend on http://localhost:8000 ...
start "DIP AI Tutor — FastAPI Backend" cmd /k ^
    "cd /d "%~dp0" && .venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000"

:: ── 4. Wait for the backend to initialise ───────────────────────────────
echo [3/4] Waiting 6 seconds for backend to initialise...
timeout /t 6 /nobreak > nul

:: ── 5. Start Gradio UI in a second terminal window ──────────────────────
echo [4/4] Starting Gradio UI on http://localhost:7860 ...
start "DIP AI Tutor — Gradio UI" cmd /k ^
    "cd /d "%~dp0" && .venv\Scripts\python.exe app\ui\interface.py"

:: ── 6. Open the browser ──────────────────────────────────────────────────
timeout /t 3 /nobreak > nul
start http://localhost:7860

echo.
echo ============================================================
echo   Both servers are launching in separate windows.
echo.
echo   Gradio UI  → http://localhost:7860
echo   FastAPI    → http://localhost:8000/docs
echo ============================================================
echo.
echo Press any key to close this launcher window.
pause > nul
