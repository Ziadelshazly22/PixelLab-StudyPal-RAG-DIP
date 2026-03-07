@echo off
title DIP AI Tutor — Quick Start
cd /d "%~dp0"

echo ============================================================
echo   DIP AI Tutor — Quick Start Launcher
echo ============================================================
echo.

:: ── 1. Verify venv exists ────────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\
    echo         Run: py -3 -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: ── 2. Check if server is already running on port 8000 ──────────────────
netstat -ano | findstr ":8000 " >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo [INFO] Backend already running on port 8000.
    echo [INFO] Opening UI directly...
    timeout /t 1 /nobreak > nul
    start http://localhost:8000/ui
    echo.
    echo ============================================================
    echo   UI   ^> http://localhost:8000/ui
    echo   Docs ^> http://localhost:8000/docs
    echo ============================================================
    pause > nul
    exit /b 0
)

:: ── 3. Start FastAPI backend (Gradio UI is mounted inside at /ui) ────────
echo [1/3] Starting backend on http://localhost:8000 ...
start "DIP AI Tutor — Backend" cmd /k "cd /d "%~dp0" && .venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000"

:: ── 4. Poll until server is ready (max 60 seconds) ──────────────────────
echo [2/3] Waiting for server to be ready...
set /a attempts=0
:poll_loop
timeout /t 2 /nobreak > nul
.venv\Scripts\python.exe -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" >nul 2>&1
if %ERRORLEVEL% == 0 goto server_ready
set /a attempts+=1
if %attempts% LSS 30 goto poll_loop
echo [ERROR] Server did not respond after 60 seconds. Check the backend window.
pause
exit /b 1

:server_ready
:: ── 5. Open browser to the Gradio UI (mounted inside FastAPI at /ui) ────
echo [3/3] Server ready — opening UI...
start http://localhost:8000/ui

echo.
echo ============================================================
echo   DIP AI Tutor is running!
echo.
echo   Chat UI    ^> http://localhost:8000/ui
echo   API Docs   ^> http://localhost:8000/docs
echo   API Status ^> http://localhost:8000/status
echo.
echo   Run "Quick Exit.bat" to stop all servers cleanly.
echo ============================================================
echo.
pause > nul
