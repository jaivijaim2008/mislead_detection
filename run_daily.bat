@echo off
REM ============================================================
REM  run_daily.bat — Missed-Lead Detector Daily Scheduler
REM  Called by Windows Task Scheduler to run the live pipeline.
REM ============================================================
REM
REM  IMPORTANT:
REM  1. Update the paths below to match YOUR system.
REM  2. Replace the App Password placeholder below.
REM  3. Test manually: double-click this file.
REM  4. Then set up Task Scheduler (see SETUP section below).
REM ============================================================

REM ---- CONFIGURATION - Update these values! -------------------
set SMTP_USER=jaivijai188@gmail.com
set SMTP_PASS=ldwc atxe rgjc eepa
set IMAP_USER=jaivijai188@gmail.com
set IMAP_PASS=ldwc atxe rgjc eepa
set SENDER_NAME=Sales Team

REM ---- Paths (update if your project is elsewhere) -----------
set PROJECT_DIR=C:\Users\HP VICTUS\Desktop\ML\missed_lead_detector
set LOG_DIR=%PROJECT_DIR%\logs
REM Use PowerShell to get a locale-independent YYYYMMDD date stamp.
REM %DATE:~..% slicing is locale-dependent and breaks in non-English Windows.
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set TODAY=%%d
set LOG_FILE=%LOG_DIR%\daily_run_%TODAY%.log

REM ---- Create log directory if needed -------------------------
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM ---- Python path (full path avoids PATH issues in Task Scheduler) ---
set PYTHON=C:\Users\HP VICTUS\AppData\Local\Programs\Python\Python311\python.exe

REM ---- Headless mode: prevent Tkinter GUI from blocking scheduled runs ---
set STREAMLIT_SERVER_HEADLESS=true

REM ---- Run the live pipeline and log output -------------------
echo ============================================================ >> "%LOG_FILE%"
echo  Missed-Lead Detector - Daily Run: %DATE% %TIME% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

cd /d "%PROJECT_DIR%"
"%PYTHON%" src/orchestrator.py --live >> "%LOG_FILE%" 2>&1

if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Pipeline exited with code %ERRORLEVEL% >> "%LOG_FILE%"
) else (
    echo  [OK] Pipeline completed successfully >> "%LOG_FILE%"
)

echo  Run complete: %DATE% %TIME% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
