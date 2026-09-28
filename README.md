# X-UBA
### Explainable User and Entity Behavior Analytics Using Heterogeneous Security Telemetry

MCA Final Year Major Project

---

## What This Is

X-UBA is an identity-risk analytics system that goes beyond simple anomaly
flagging. It integrates heterogeneous IAM telemetry (authentication logs,
privilege-change logs, resource-access logs, and identity metadata) into a
unified risk-scoring pipeline that is:

- **Multi-dimensional** — fuses behavioral, temporal, and privilege signals
- **Explainable** — every risk score is decomposed via SHAP into contributing factors
- **Impact-aware** — models what an attacker could reach if an identity is compromised (blast radius)
- **Actionable** — simulates "what-if" security actions and ranks them by predicted risk reduction

## Project Structure

```
X-UBA/
├── README.md
├── requirements.txt
├── setup_environment.ps1          # one-command environment setup (Windows)
├── data/
│   ├── raw/                       # 6 heterogeneous telemetry tables
│   │   ├── identities.csv
│   │   ├── login_events.csv
│   │   ├── privilege_changes.csv
│   │   ├── resource_access.csv
│   │   ├── resources.csv
│   │   └── systems.csv
│   └── processed/                 # engineered features + train/test split
│       ├── identity_features.csv
│       ├── train_features.csv
│       ├── test_features.csv
│       └── README_data_summary.txt
├── src/
│   ├── data_generation/
│   │   └── generate_dataset.py    # Phase 0: synthetic telemetry generator
│   ├── modeling/
│   │   └── train_models.py        # Phase 1: XGBoost + SHAP training
│   ├── anomaly_detection/
│   │   └── train_isolation_forest.py  # Phase 2: unsupervised anomaly detection
│   ├── explainability/            # SHAP helpers currently live in modeling/
│   ├── graph_analysis/            # Phase 5: NetworkX privilege graph + blast radius (upcoming)
│   ├── simulator/                 # Phase 6-7: counterfactual + attack simulator (upcoming)
│   └── dashboard/                 # Phase 8: Flask web dashboard (upcoming)
│       └── templates/
├── models/                        # trained model artifacts (.joblib)
├── reports/                       # metrics, SHAP importances, figures
│   └── figures/
├── notebooks/                     # exploratory analysis (optional)
└── docs/                          # literature review, architecture diagrams, final report
```

## Architecture (Planned Full Pipeline)

```
Heterogeneous Telemetry (6 tables)
            |
   Feature Engineering
            |
   ┌────────┼────────┐
   |        |         |
XGBoost  Isolation  K-Means
(risk)    Forest    (behavior
            |        clusters)
   └────────┼────────┘
            |
    Temporal Risk Trajectory
            |
        Risk Fusion
            |
   ┌────────┴────────┐
   |                  |
SHAP              Privilege Graph
Explainability    (NetworkX)
   |                  |
"WHY risky?"    Attack Path + Blast Radius
                       |
              ┌────────┴────────┐
              |                 |
        Counterfactual     ATTACK SIMULATOR
        "what if?"          "what could happen?"
              └────────┬────────┘
                        |
              Security Dashboard (Flask)
```

## Current Status

| Phase | Module | Status |
|---|---|---|
| 0 | Synthetic heterogeneous telemetry generation (10,000 identities) | ✅ Done |
| 1 | XGBoost risk classification (is_anomaly, risk_level, threat_type) | ✅ Done |
| 1b | SHAP explainability | ✅ Done |
| 2 | Isolation Forest anomaly detection (unsupervised, compared vs XGBoost) | ✅ Done |
| 3 | K-Means behavioral clustering (4 named clusters) | ✅ Done |
| 4 | Temporal risk trajectory (early-warning detection) | ✅ Done |
| 5 | NetworkX privilege graph + risk-weighted blast radius | ✅ Done |
| 6 | Counterfactual risk-reduction engine | ✅ Done |
| 7 | AI Identity Attack Simulator (flagship feature) | ✅ Done |
| 8a | Separation of Duties (SOD) violation detection | ✅ Done |
| 8b | Compliance gap analysis (NIST AC-2 / GDPR Article 32) | ✅ Done |
| 8c | Organizational (department-level) anomaly detection | ✅ Done |
| 9a | **Multi-Signal Risk Fusion** (research gap closure) | ✅ Done |
| 9b | **SHAP Explanation Fidelity Quantification** (research gap closure) | ✅ Done |
| 9c | **Graph-Structural Peer-Group Anomaly Detection** (research gap closure) | ✅ Done |
| 10 | Full Flask dashboard (UI integration of all modules above) | ⏳ Next |
| 11 | Final report + literature comparison | ⏳ Planned |

## Research Gap Closure (Literature-Grounded Additions)

Three gaps identified from actual 2025-2026 literature (not assumed):

**Source:** Tenali & Potnuri, "A Comprehensive Review of Recent Challenges
and Emerging Trends in Malicious Insider Threat Detection Using Machine
Learning-Based Methods," Springer LNNS vol. 2019, ICICC 2026 — names
**multimodal fusion**, **graph-based modeling**, and **explainability
benchmarking through metrics of explanation fidelity** as underexploited
emerging trends. Corroborated by multiple 2025-2026 GNN-for-IAM papers
(e.g. arXiv:2512.10280) explicitly criticizing systems that score
identities "in isolation, without modeling relational dependencies
among entities."

### Gap 1: Multi-Signal Risk Fusion
X-UBA's own architecture diagram always showed a "Risk Fusion" stage
combining XGBoost + Isolation Forest + Temporal + Graph signals — this
was never actually implemented until now. A Logistic Regression
meta-model stacks all 4 signals into one calibrated score.
**Result: Fusion F1=0.882 vs best individual signal (XGBoost alone)
F1=0.862 — a genuine, measured +2.0 point improvement**, not asserted.
Temporal trend is nearly uncorrelated with the other 3 signals (r<0.03),
confirming it carries genuinely complementary information despite being
the weakest standalone predictor (F1=0.35) — exactly the diversity that
makes ensemble fusion work.

### Gap 2: SHAP Explanation Fidelity Quantification
Most UEBA/insider-threat systems show SHAP explanations but never verify
they're faithful to the model's real reasoning. Implemented a
deletion-based fidelity test (comparable to the ERASER benchmark's
"comprehensiveness" metric): remove the top-5 SHAP-ranked features vs.
5 random features, measure the drop in predicted probability.
**Result: top-5 SHAP features cause a 74x larger probability drop than
random features (0.393 vs 0.007), with 83.2% of identities showing high
fidelity (ratio ≥1.5x)** — a genuine, quantified validation that X-UBA's
explanations are measurably faithful, not just plausible-looking.

### Gap 3: Graph-Structural Peer-Group Anomaly Detection
Phase 5's privilege graph was previously used only for blast-radius
*impact*, never for *detection*. Built an identity-identity graph from
shared critical-resource access, ran Louvain community detection, and
flagged identities whose risk deviates sharply from their structural
peer group. **Honest result: on this synthetic dataset, 0 of 154
structural outliers were "novel" (all were already flagged CRITICAL by
Phase 1)** — because the data generator's risk-weighted event sampling
correlates behavioral risk with critical-resource exposure almost
deterministically. This is reported transparently as a genuine
limitation of validating relational methods on synthetic data, not
hidden or forced into a false positive result — and is itself a
legitimate, defensible methodological finding for the final report.

## Setup

See `setup_environment.ps1` for a one-command environment setup (Windows,
PowerShell), or run manually, in this exact order (later modules depend on
earlier ones' outputs):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src\data_generation\generate_dataset.py
python src\modeling\train_models.py
python src\anomaly_detection\train_isolation_forest.py
python src\clustering\train_kmeans_clustering.py
python src\temporal_analysis\temporal_risk_trajectory.py
python src\graph_analysis\build_privilege_graph.py
python src\simulator\counterfactual_engine.py
python src\simulator\attack_simulator.py
python src\compliance\sod_violations.py
python src\compliance\compliance_gap_analysis.py
python src\organizational_analysis\org_anomaly_detection.py
python src\risk_fusion\risk_fusion_engine.py
python src\explainability\explanation_fidelity.py
python src\graph_analysis\graph_structural_anomaly.py
```

## Current Model Performance (Phase 1 + Phase 2)

| Model | Type | Accuracy | F1 |
|---|---|---|---|
| is_anomaly | XGBoost (supervised) | 94.1% | 92.2% (macro) |
| is_anomaly | Isolation Forest (unsupervised) | 84.4% | 69.2% |
| risk_level (4-class) | XGBoost (supervised) | 84.9% | 83.2% (macro) |
| threat_type (6-class) | XGBoost (supervised) | 92.9% | 66.0%* (macro) |

The Isolation Forest vs XGBoost gap (69.2% vs 92.2% F1 on the *identical*
feature set) is itself a reportable finding: it quantifies how much
labeled training data improves detection, while the unsupervised model
still provides a label-free safety net for novel anomaly types the
supervised model was never trained on.

\* Macro-F1 is lower than accuracy for `threat_type` because rare attack
classes (BRUTE_FORCE, DATA_EXFILTRATION, INSIDER_THREAT) have limited
support in the training data — a known and documented class-imbalance
limitation, addressed as future work via SMOTE/class-weighting.

## Phase 3-7 Results Summary

**Phase 3 (Clustering):** 4 named behavioral clusters, e.g. "Stale Admin +
MFA-Gap + High-Risk" (1,408 identities, 74% CRITICAL/HIGH) vs "Low-Risk /
Baseline" (4,389 identities, 84% LOW risk).

**Phase 4 (Temporal):** 1,966 identities flagged with an EARLY_WARNING —
risk rising sharply even when current absolute risk isn't yet CRITICAL.

**Phase 5 (Blast Radius):** Validated — average blast radius score rises
monotonically with actual risk level (32.5 → 43.2 → 58.2 → 70.0), with a
0.693 correlation to the ground-truth threat score.

**Phase 6 (Counterfactual):** "Reduce System Footprint" is the single most
effective intervention (average -9.22 points), independently matching
Phase 1's top SHAP-ranked features — convergent validation between
explainability and actionability.

**Phase 7 (Attack Simulator):** Fully integrates Phases 1, 3, 4, 5, 6 into
one per-identity WHY → WHAT → IMPACT → ACTION report.

**Phase 8a (SOD Violations):** 1,057 identities (10.6%) flagged across 5
rule types; most common is "Admin Without MFA" (587 identities).

**Phase 8b (Compliance Gaps):** 78% of identities have at least one NIST/GDPR
gap - reflecting the identity-sprawl problem this project targets, not a
detection error. Average compliance score: 71.3/100.

**Phase 8c (Org Anomalies):** Departments show only minor risk variation
(26.8-28.6 avg) because `department` is generated independently of the
actual risk-driving factors (privilege, admin status, inactivity) -
this is a deliberate validation that the risk model doesn't leak
department as a spurious signal (department is intentionally excluded
from Phase 1's feature set). Real deployments with genuine department-
correlated access patterns (e.g. Finance month-end spikes) would show
sharper organizational anomalies; injecting such correlated patterns
into the synthetic generator is noted as future work.

Full metrics for every phase: see individual `reports/phaseN_*_summary.txt` files.

## Novelty Statement

Existing identity/UEBA research largely stops at *detecting* anomalous or
risky identities. X-UBA extends this into **impact assessment and proactive
mitigation**: for any identity, the system explains *why* it is risky (SHAP),
estimates *what* could happen if it were compromised (graph-based blast
radius), and simulates *which* security action produces the greatest
predicted risk reduction (counterfactual analysis) — a
**Detect → Explain → Simulate → Mitigate** workflow rather than detection alone.