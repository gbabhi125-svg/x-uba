@echo off
setlocal enabledelayedexpansion

REM ═══════════════════════════════════════════════════════════════════════
REM   MCA CAPSTONE — FRESH MACHINE SETUP (Windows)
REM   Assumes: brand new machine, nothing installed except (maybe) Python.
REM   This script creates an isolated virtual environment so it CANNOT
REM   conflict with anything else on the machine (avoids the dependency
REM   conflict mess you hit last time with global pip installs).
REM ═══════════════════════════════════════════════════════════════════════

echo.
echo ============================================================
echo   MCA CAPSTONE PROJECT - FRESH SETUP
echo ============================================================
echo.

REM ── Step 1: Check Python exists ─────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python not found on this machine.
    echo.
    echo     Download and install Python 3.10 or 3.11 from:
    echo     https://www.python.org/downloads/
    echo.
    echo     IMPORTANT: During install, check the box that says
    echo     "Add python.exe to PATH" - then re-run this script.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version') do set PYVER=%%v
echo [OK] Python found: %PYVER%
echo.

REM ── Step 2: Confirm required files are present ──────────────────────────
if not exist "requirements.txt" (
    echo [X] requirements.txt not found in this folder.
    echo     Make sure you copied ALL project files into this folder first.
    pause
    exit /b 1
)
if not exist "generate_full_dataset.py" (
    echo [X] generate_full_dataset.py not found in this folder.
    pause
    exit /b 1
)
if not exist "train_models.py" (
    echo [X] train_models.py not found in this folder.
    pause
    exit /b 1
)
echo [OK] All required project files found.
echo.

REM ── Step 3: Create isolated virtual environment ─────────────────────────
if exist "venv" (
    echo [OK] Virtual environment already exists - reusing it.
) else (
    echo Creating virtual environment (venv)...
    python -m venv venv
    if errorlevel 1 (
        echo [X] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)
echo.

REM ── Step 4: Install dependencies INSIDE the venv only ────────────────────
echo Installing dependencies into the isolated environment...
echo (this will NOT touch or conflict with anything else on your machine)
echo.
venv\Scripts\python.exe -m pip install --upgrade pip -q
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [X] Dependency installation failed. Check the error above.
    pause
    exit /b 1
)
echo [OK] All dependencies installed cleanly inside venv\
echo.

REM ── Step 5: Generate the dataset (Phase 0) ───────────────────────────────
echo ============================================================
echo   PHASE 0: Generating dataset (10,000 identities)
echo ============================================================
venv\Scripts\python.exe generate_full_dataset.py
if errorlevel 1 (
    echo [X] Data generation failed.
    pause
    exit /b 1
)
echo.

REM ── Step 6: Train models (Phase 1) ───────────────────────────────────────
echo ============================================================
echo   PHASE 1: Training XGBoost models + SHAP explainability
echo ============================================================
venv\Scripts\python.exe train_models.py
if errorlevel 1 (
    echo [X] Model training failed.
    pause
    exit /b 1
)
echo.

REM ── Done ──────────────────────────────────────────────────────────────
echo ============================================================
echo   [OK] SETUP COMPLETE
echo ============================================================
echo.
echo   data\raw\           - 6 raw IAM tables (10,000 identities)
echo   data\processed\     - engineered features + train/test split
echo   models\             - trained XGBoost models + SHAP explainer
echo   reports\            - accuracy/precision/recall/F1 + SHAP importances
echo.
echo   From now on, to run any script, use:
echo     venv\Scripts\python.exe your_script.py
echo.
echo   (This keeps everything isolated in the venv folder.)
echo.
pause