# =======================================================================
#   X-UBA - MCA MAJOR PROJECT - ENVIRONMENT SETUP (PowerShell / VS Code)
#   Run from the project ROOT folder:   .\setup_environment.ps1
#   If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1
# =======================================================================

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Fail($msg) {
    Write-Host "[X] $msg" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================"
Write-Host "  X-UBA - MCA Major Project - Environment Setup"
Write-Host "============================================================"
Write-Host ""

# -- Step 1: Python 3.11+ ---------------------------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Fail "Python not found. Install Python 3.11 - 3.14 from python.org and tick 'Add python.exe to PATH'."
}
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { Fail "Python 3.11 or newer is required (found: $(python --version 2>&1))." }
Write-Host "[OK] $(python --version 2>&1)"

# -- Step 2: project structure --------------------------------------------
foreach ($f in @("requirements.txt", "run_pipeline.py", "src\data_generation\generate_dataset.py")) {
    if (-not (Test-Path $f)) { Fail "$f not found. Run this script from the X-UBA project root." }
}
Write-Host "[OK] Project structure verified."

# -- Step 3: virtual environment ------------------------------------------
if (Test-Path "venv\Scripts\python.exe") {
    Write-Host "[OK] Virtual environment already exists - reusing it."
} else {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." }
    Write-Host "[OK] Virtual environment created."
}
$py = "venv\Scripts\python.exe"

# -- Step 4: dependencies --------------------------------------------------
Write-Host "Installing dependencies (isolated inside venv\)..."
& $py -m pip install --upgrade pip -q
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed - see the error above." }
Write-Host "[OK] Dependencies installed."
Write-Host ""

# -- Step 5: full pipeline (stops on the first failing module) -------------
& $py run_pipeline.py
if ($LASTEXITCODE -ne 0) { Fail "The pipeline stopped with an error - see the message above." }

Write-Host ""
Write-Host "============================================================"
Write-Host "  [OK] SETUP COMPLETE - ALL 14 MODULES RAN SUCCESSFULLY" -ForegroundColor Green
Write-Host "============================================================"
Write-Host ""
Write-Host "  Dashboard:   venv\Scripts\python.exe src\dashboard\app.py"
Write-Host "               then open http://127.0.0.1:5000"
Write-Host "  One module:  venv\Scripts\python.exe run_pipeline.py --only 8"
Write-Host "  One report:  venv\Scripts\python.exe src\simulator\attack_simulator.py --identity U00042"
Write-Host ""
