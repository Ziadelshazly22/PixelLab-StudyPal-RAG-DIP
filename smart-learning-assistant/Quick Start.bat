@echo off
chcp 65001 >nul 2>&1
title DIP AI Tutor - Quick Start
cd /d "%~dp0"

echo.
echo ============================================================
echo   DIP AI Tutor  -  Quick Start Launcher
echo ============================================================
echo.

:: --------------------------------------------------------
:: 1. Verify virtual environment exists
:: --------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo.
    echo   To create it, open a terminal here and run:
    echo     py -3 -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: --------------------------------------------------------
:: 2. Check if server is already running on port 8000
:: --------------------------------------------------------
netstat -ano | findstr ":8000 " >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo [INFO] Backend is already running on port 8000.
    echo [INFO] Opening the UI now ...
    echo.
    start http://localhost:8000/ui
    echo ============================================================
    echo   Chat UI   -^>  http://localhost:8000/ui
    echo   API Docs  -^>  http://localhost:8000/docs
    echo   Status    -^>  http://localhost:8000/status
    echo ============================================================
    echo.
    echo   Press any key to close this window.
    pause >nul
    exit /b 0
)

:: --------------------------------------------------------
:: 3. Launch FastAPI backend in a new window
::    /D sets the working directory for the new window
:: --------------------------------------------------------
echo [1/3] Starting backend server ...
echo       (A new terminal window will open -- do not close it)
echo.
start "DIP AI Tutor - Backend" /D "%~dp0" cmd /k ".venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000"

:: --------------------------------------------------------
:: 4. Poll until server responds  (max 90 s, 2 s intervals)
::    Using ping for delay -- immune to Ctrl+C interrupts
:: --------------------------------------------------------
echo [2/3] Waiting for server to be ready ...
echo       (this takes 10-30 seconds on first start)
echo.
set /a attempts=0

:poll_loop
ping -n 3 127.0.0.1 >nul
curl -sf --connect-timeout 2 --max-time 3 http://127.0.0.1:8000/health >nul 2>&1
if %ERRORLEVEL% == 0 goto server_ready
set /a attempts+=1
set /a elapsed=attempts*2
echo   ... still waiting (%elapsed%s elapsed, max 90s) ...
if %attempts% LSS 45 goto poll_loop

echo.
echo [ERROR] Server did not respond after 90 seconds.
echo.
echo   Possible causes:
echo     - A Python import error in main.py  (check the backend terminal)
echo     - Missing .env file or GROQ_API_KEY
echo     - ChromaDB not yet built  (run scripts\run_ingestion.py first)
echo.
pause
exit /b 1

:server_ready
:: --------------------------------------------------------
:: 5. Server is ready -- open browser and show links
:: --------------------------------------------------------
echo [3/3] Server is ready!
echo.
start http://localhost:8000/ui

echo ============================================================
echo.
echo   DIP AI Tutor is running!
echo.
echo   Chat UI    -^>  http://localhost:8000/ui
echo   API Docs   -^>  http://localhost:8000/docs
echo   API Status -^>  http://localhost:8000/status
echo.
echo   If the browser did not open automatically, copy and
echo   paste the Chat UI link above into your browser.
echo.
echo   To stop all servers cleanly, run:  Quick Exit.bat
echo.
echo ============================================================
echo.
echo   This launcher window will close in 10 seconds...
echo   (The backend server window must stay open while you use the app)
echo.
timeout /t 10 /nobreak >nul
exit /b 0

