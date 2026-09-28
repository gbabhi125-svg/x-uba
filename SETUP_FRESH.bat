@echo off
setlocal

REM ======================================================================
REM   X-UBA - MCA MAJOR PROJECT - FRESH MACHINE SETUP (Windows)
REM   Double-click this file from the X-UBA project folder.
REM   Creates an isolated virtual environment (venv\), installs the
REM   dependencies into it, runs all analysis modules, and tells you
REM   how to open the dashboard. Nothing is installed globally.
REM ======================================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo   X-UBA - MCA MAJOR PROJECT - FRESH SETUP
echo ============================================================
echo.

REM -- Step 1: Python 3.11+ -------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python not found on this machine.
    echo     Install Python 3.11 - 3.14 from https://www.python.org/downloads/
    echo     and tick "Add python.exe to PATH" during install, then re-run this file.
    pause
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo [X] Python 3.11 or newer is required. Found:
    python --version
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version') do echo [OK] Python %%v found
echo.

REM -- Step 2: project files ----------------------------------------------
if not exist "requirements.txt" goto missing
if not exist "run_pipeline.py" goto missing
if not exist "src\data_generation\generate_dataset.py" goto missing
echo [OK] Project files found.
echo.

REM -- Step 3: virtual environment ----------------------------------------
if exist "venv\Scripts\python.exe" (
    echo [OK] Virtual environment already exists - reusing it.
) else (
    echo Creating virtual environment venv\ ...
    python -m venv venv
    if errorlevel 1 (
        echo [X] Failed to create the virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)
echo.

REM -- Step 4: dependencies (inside venv only) ----------------------------
echo Installing dependencies into venv\ (this does not touch anything else)...
venv\Scripts\python.exe -m pip install --upgrade pip -q
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [X] Dependency installation failed. Check the error above.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.
echo.

REM -- Step 5: full pipeline ----------------------------------------------
venv\Scripts\python.exe run_pipeline.py
if errorlevel 1 (
    echo.
    echo [X] The pipeline stopped with an error - see the message above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   [OK] SETUP COMPLETE - ALL MODULES RAN SUCCESSFULLY
echo ============================================================
echo.
echo   Open the dashboard:
echo     venv\Scripts\python.exe src\dashboard\app.py
echo   then browse to http://127.0.0.1:5000
echo.
pause
exit /b 0

:missing
echo [X] Project files not found. Run this file from the X-UBA project folder
echo     (the folder that contains requirements.txt and run_pipeline.py).
pause
exit /b 1
