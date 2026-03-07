@echo off
title DIP AI Tutor — Quick Exit
cd /d "%~dp0"

echo ============================================================
echo   DIP AI Tutor — Shutdown and Clean Session
echo ============================================================
echo.

:: ── 1. Kill FastAPI / Uvicorn on port 8000 ───────────────────────────────
echo [1/3] Stopping backend (port 8000)...
set /a killed8000=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000 " 2^>nul') do (
    if not "%%p"=="0" if not "%%p"=="PID" (
        taskkill /PID %%p /F >nul 2>&1
        set /a killed8000+=1
    )
)
if %killed8000% GTR 0 (
    echo         Done. (%killed8000% process(es) terminated)
) else (
    echo         Not running.
)

:: ── 2. Kill standalone Gradio on port 7860 (if launched separately) ─────
echo [2/3] Stopping standalone Gradio (port 7860)...
set /a killed7860=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":7860 " 2^>nul') do (
    if not "%%p"=="0" if not "%%p"=="PID" (
        taskkill /PID %%p /F >nul 2>&1
        set /a killed7860+=1
    )
)
if %killed7860% GTR 0 (
    echo         Done. (%killed7860% process(es) terminated)
) else (
    echo         Not running.
)

:: ── 3. Clear Python and pytest cache ────────────────────────────────────
echo [3/3] Clearing Python cache...
for /r "%~dp0" %%d in (__pycache__) do (
    if exist "%%d" (
        rd /s /q "%%d" >nul 2>&1
    )
)
if exist "%~dp0.pytest_cache" rd /s /q "%~dp0.pytest_cache" >nul 2>&1
echo         Done.

echo.
echo ============================================================
echo   All sessions stopped. Cache cleared.
echo   Ready for a fresh start — run "Quick Start.bat" again.
echo ============================================================
echo.
pause > nul
