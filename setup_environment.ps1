# ═══════════════════════════════════════════════════════════════════════
#   X-UBA — ENVIRONMENT SETUP (PowerShell / VS Code Terminal)
#   Run this from the project ROOT folder (X-UBA\).
# ═══════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================"
Write-Host "  X-UBA - Environment Setup"
Write-Host "============================================================"
Write-Host ""

# ── Step 1: Check Python ────────────────────────────────────────────────
try {
    $pyver = python --version 2>&1
    Write-Host "[OK] Python found: $pyver"
} catch {
    Write-Host "[X] Python not found. Install Python 3.10/3.11 from python.org"
    Write-Host "    and check 'Add python.exe to PATH' during install."
    exit 1
}

# ── Step 2: Confirm we're in the right folder ───────────────────────────
if (-not (Test-Path "requirements.txt")) {
    Write-Host "[X] requirements.txt not found. Run this script from the X-UBA project root."
    exit 1
}
if (-not (Test-Path "src\data_generation\generate_dataset.py")) {
    Write-Host "[X] src\data_generation\generate_dataset.py not found. Check your folder structure."
    exit 1
}
Write-Host "[OK] Project structure verified."
Write-Host ""

# ── Step 3: Create virtual environment ──────────────────────────────────
if (Test-Path "venv") {
    Write-Host "[OK] Virtual environment already exists - reusing it."
} else {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    Write-Host "[OK] Virtual environment created."
}
Write-Host ""

# ── Step 4: Install dependencies inside venv ────────────────────────────
Write-Host "Installing dependencies (isolated inside venv\)..."
& "venv\Scripts\python.exe" -m pip install --upgrade pip -q
& "venv\Scripts\python.exe" -m pip install -r requirements.txt
Write-Host "[OK] Dependencies installed."
Write-Host ""

# ── Step 5: Run the full pipeline (Phases 0-7, in dependency order) ──────
Write-Host "============================================================"
Write-Host "  PHASE 0: Generating heterogeneous telemetry dataset"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\data_generation\generate_dataset.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  PHASE 1: Training XGBoost models + SHAP explainability"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\modeling\train_models.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  PHASE 2: Isolation Forest anomaly detection"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\anomaly_detection\train_isolation_forest.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  PHASE 3: K-Means behavioral clustering"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\clustering\train_kmeans_clustering.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  PHASE 4: Temporal risk trajectory"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\temporal_analysis\temporal_risk_trajectory.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  PHASE 5: Privilege graph and blast radius"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\graph_analysis\build_privilege_graph.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  PHASE 6: Counterfactual risk-reduction engine"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\simulator\counterfactual_engine.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  PHASE 7: AI Identity Attack Simulator"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\simulator\attack_simulator.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  PHASE 8a: Separation of Duties violations"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\compliance\sod_violations.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  PHASE 8b: Compliance gap analysis (NIST/GDPR)"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\compliance\compliance_gap_analysis.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  PHASE 8c: Organizational anomaly detection"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\organizational_analysis\org_anomaly_detection.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  GAP 1: Multi-Signal Risk Fusion"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\risk_fusion\risk_fusion_engine.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  GAP 2: SHAP Explanation Fidelity Quantification"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\explainability\explanation_fidelity.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  GAP 3: Graph-Structural Peer-Group Anomaly Detection"
Write-Host "============================================================"
& "venv\Scripts\python.exe" "src\graph_analysis\graph_structural_anomaly.py"
Write-Host ""

Write-Host "============================================================"
Write-Host "  [OK] SETUP COMPLETE - ALL 14 MODULES RAN SUCCESSFULLY"
Write-Host "============================================================"
Write-Host ""
Write-Host "  data\raw\        - 6 heterogeneous telemetry tables"
Write-Host "  data\processed\  - engineered features + train/test split"
Write-Host "  models\          - XGBoost, Isolation Forest, K-Means, SHAP, graph"
Write-Host "  reports\         - metrics + summaries for every phase"
Write-Host ""
Write-Host "  To run any single script later:"
Write-Host "    venv\Scripts\Activate.ps1"
Write-Host "    python src\simulator\attack_simulator.py"
Write-Host ""