# X-UBA
### Explainable User and Entity Behavior Analytics Using Heterogeneous Security Telemetry

**MCA Major Project**

---

## What this is

X-UBA is an identity-risk analytics system that goes beyond flagging anomalies.
It fuses heterogeneous IAM telemetry (authentication, privilege-change and
resource-access logs plus identity metadata) into one pipeline that, for every
identity, answers:

| Question | Module |
|---|---|
| **WHY** is it risky? | XGBoost risk models + SHAP explanations |
| **WHO** does it behave like? | K-Means behavioural peer groups |
| **WHEN** is it getting worse? | Temporal risk trajectory |
| **WHAT** could an attacker do? | Kill-chain attack simulator |
| **IMPACT** — how far could it spread? | NetworkX privilege graph + blast radius |
| **ACTION** — what should we do? | Counterfactual "what-if" engine |

plus separation-of-duties checks, NIST SP 800-53 / GDPR Art. 32 compliance
gaps, department-level anomalies, three research-gap experiments, a
cross-layer **quiet-risk** analysis, a fixed-budget **remediation optimizer**,
and an interactive web dashboard over all of it.

## Quick start (Windows, VS Code)

Requires Python **3.11 – 3.14** (tick *Add python.exe to PATH* when installing).

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

python run_pipeline.py            # all 17 modules, ~2 minutes
python src\dashboard\app.py       # then open http://127.0.0.1:5000
```

Or double-click `SETUP_FRESH.bat` (or run `.\setup_environment.ps1`), which does
the environment + pipeline in one go. On Linux/macOS use `source venv/bin/activate`.

If PowerShell blocks `Activate.ps1`: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Other useful commands:

```powershell
python run_pipeline.py --list                  # list the 17 steps
python run_pipeline.py --only 9                # re-run one step (9 = attack simulator)
python run_pipeline.py --from 5                # resume from a step
python src\simulator\attack_simulator.py --identity U05473   # one identity's full report
```

`run_pipeline.py` stops at the first failing module and exits non-zero, so a
failure can never be reported as success.

## Reproducibility

- Fixed random seed (42) in every module.
- `requirements.txt` pins exact library versions.
- Verified: on the same operating system the complete pipeline produces
  **byte-identical outputs on Python 3.11, 3.12, 3.13 and 3.14**, and across
  repeated runs.
- The generated **data is identical on Windows and Linux**. Model scores can
  differ in the third decimal between operating systems (XGBoost uses the
  platform's maths library), e.g. `is_anomaly` accuracy 0.941 on Windows vs
  0.9435 on Linux. Quote the numbers from your own `reports/` folder — those are
  what the dashboard shows. The tables below are from a Linux run.

## Project structure

```
X-UBA/
├── run_pipeline.py                 # one command for all 17 modules
├── requirements.txt                # pinned versions
├── SETUP_FRESH.bat / setup_environment.ps1
├── src/
│   ├── common.py                   # shared loading, encoding, risk index
│   ├── data_generation/generate_dataset.py          Phase 0
│   ├── modeling/train_models.py                     Phase 1
│   ├── modeling/model_validation.py                 Phase 1b (CV, ablation, class weights)
│   ├── anomaly_detection/train_isolation_forest.py  Phase 2
│   ├── clustering/train_kmeans_clustering.py        Phase 3
│   ├── temporal_analysis/temporal_risk_trajectory.py Phase 4
│   ├── graph_analysis/build_privilege_graph.py      Phase 5
│   ├── graph_analysis/reach.py                      shared attacker-reach model
│   ├── simulator/counterfactual_engine.py           Phase 6
│   ├── simulator/interventions.py                   shared security actions
│   ├── simulator/killchain.py                       shared kill-chain model
│   ├── simulator/attack_simulator.py                Phase 7
│   ├── compliance/sod_violations.py                 Phase 8a
│   ├── compliance/compliance_gap_analysis.py        Phase 8b
│   ├── organizational_analysis/org_anomaly_detection.py Phase 8c
│   ├── risk_fusion/risk_fusion_engine.py            Gap 1
│   ├── explainability/explanation_fidelity.py       Gap 2
│   ├── graph_analysis/graph_structural_anomaly.py   Gap 3
│   ├── risk_fusion/quiet_risk.py                    Ext 1 (cross-layer disagreement)
│   ├── simulator/remediation_optimizer.py           Ext 2 (fixed-budget remediation)
│   └── dashboard/                                   Phase 9: Flask API + single-page app
│       ├── app.py                                   JSON API, live SHAP / what-if / remediation
│       ├── templates/index.html
│       └── static/ (css, js, bundled Chart.js + D3 -> works offline)
├── data/raw, data/processed        # generated (not committed)
├── models/                         # generated
└── reports/                        # generated: *_summary.txt, *_metrics.json, CSVs
```

## Dashboard

`python src\dashboard\app.py` → http://127.0.0.1:5000 (works offline; Chart.js
and D3 are bundled). Nine tabs, all driven by a JSON API — no number is
hard-coded in the page:

| Tab | What it shows |
|---|---|
| Overview | KPIs, predicted risk / threat mix, behavioural clusters, top attack targets |
| Identity Explorer | search / filter / sort all 10,000 identities; click for the deep-dive |
| Identity deep-dive | **live SHAP** from the trained model, class probabilities, monthly activity, kill chain + attack path, SoD evidence, NIST/GDPR controls, **live what-if** (MFA, privilege, dormancy, systems) and **one-click remediation** re-scored by the model with blast radius re-computed on the privilege graph |
| Privilege Graph | department→system access graph and any identity's ego-network (D3, drag/zoom) |
| Model Performance | per-class metrics, confusion matrix, 5-fold CV, feature-group ablation, class-weighted threat model |
| Compliance & Policy | SoD rules and severities, control gaps, worst offenders |
| Organization | department comparison (chi-square), within-department peer outliers |
| Quiet Risk | behavioural vs structural scatter, quadrants, top quiet-risk identities |
| Remediation Optimizer | naive vs impact-aware vs balanced plans under a 200-fix budget |
| Research Results | the three research gaps + explanation-action agreement, full report texts |

## Dataset

Synthetic heterogeneous telemetry for **10,000 identities**, generated
deterministically (seed 42):

| Table | Rows |
|---|---|
| systems | 15 (AD, Azure AD, AWS IAM, Okta, PROD_DB, Data Lake, ...) |
| resources | 428 |
| identities | 10,000 (risk: LOW 4,312 / MEDIUM 2,664 / HIGH 1,711 / CRITICAL 1,313) |
| login_events | 300,000 |
| privilege_changes | 30,000 |
| resource_access | 150,000 |

Aggregated into 30 identity features; stratified 80/20 split (8,000 train /
2,000 test).

## Results (held-out test set unless stated)

### Phase 1–2: detection

| Model | Accuracy | F1 |
|---|---|---|
| XGBoost `is_anomaly` | 94.4% | 92.5% (macro), 88.8% (anomaly class) |
| XGBoost `risk_level` (4-class) | 85.4% | 83.9% (macro) |
| XGBoost `threat_type` (6-class) | 92.6% | 63.8% (macro)* |
| Isolation Forest (unsupervised) | 84.4% | 69.2% (anomaly class) |

Like-for-like (anomaly class), labelled data lifts F1 from 0.692 to 0.888.
\* Rare threat classes (INSIDER_THREAT: 30 test cases) have low recall; class
imbalance is a known limitation.

Downstream modules use **5-fold out-of-fold** scores, so no identity is scored
by a model that saw its label.

### Phases 3–8

| Phase | Headline result |
|---|---|
| 3 K-Means | 4 unlabelled clusters; true-anomaly rate ranges from 94.9% ("Stale + Admin-Heavy + MFA-Gap ...", 1,313 ids) to 0.1% ("Low-Activity + Low-Risk", 3,959 ids). Silhouette 0.14 (overlapping groups). |
| 4 Temporal | 574 early warnings. Activity **level** AUC 0.963; activity **trend** AUC 0.495 (see limitations). |
| 5 Blast radius | Graph of 10,443 nodes / 163,798 edges. Mean blast radius rises monotonically LOW 20.5 → MEDIUM 30.3 → HIGH 50.2 → CRITICAL 64.4; Pearson r = 0.578 with ground truth. |
| 6 Counterfactual | Most effective action: *Reduce System Footprint* (−9.1 points on average over 2,983 HIGH/CRITICAL identities). Action effectiveness agrees with SHAP feature ranking (Spearman 0.975). |
| 7 Attack simulator | Label-free kill-chain ranking: AUC 0.867; the top-100 simulated targets are 100% true anomalies. |
| 8a SOD | 1,936 identities (19.4%) violate at least one of 6 rules (e.g. Admin Without MFA: 587). Flagged identities are 60.3% anomalous vs 17.0% otherwise. |
| 8b Compliance | 59.1% of identities have ≥1 NIST/GDPR gap; average score 83.9/100 (LOW 95.6 → CRITICAL 62.3). |
| 8c Organisation | Department does not predict anomaly (chi-square p = 0.83). Within-department peer outliers: 783, of which 86.5% are true anomalies (base rate 25.4%). |

### Phase 1b: model validation

| Model | 5-fold CV accuracy | 5-fold CV macro-F1 |
|---|---|---|
| `is_anomaly` | 94.4% ± 0.5 | 0.926 ± 0.007 |
| `risk_level` | 84.2% ± 0.9 | 0.825 ± 0.010 |
| `threat_type` | 93.1% ± 0.2 | 0.664 ± 0.020 |

Feature-group ablation (risk_level, test macro-F1 drop when a source is
removed): authentication logs −0.056, identity posture −0.054, resource-access
logs −0.027, privilege-change logs −0.007, derived signals −0.008 — every
telemetry source contributes. Class-weighted `threat_type` raises macro-F1
0.638 → 0.677; INSIDER_THREAT F1 0.10 → 0.33.

### Extensions

**Quiet risk (cross-layer disagreement).** Behavioural layer (cross-validated
model risk) vs structural layer (blast radius + SoD severity + compliance gap).
**411 identities** are rated LOW/MEDIUM by the behavioural model while being in
the structural top 20% — 99% can reach a CRITICAL resource and the most common
reason is *Admin Without MFA* (190). A behaviour-only UEBA system would rank
them safe.

**Remediation optimizer (budget = 200 fixes).**

| Strategy | Model-risk reduction | Expected attack risk removed | Critical paths removed |
|---|---|---|---|
| Naive (behaviour-only) | **7,536** | 1,219 (3.2%) | 941 |
| Impact-aware (graph + kill chain) | 1,256 | **6,138 (16.3%)** | **5,099** |
| Balanced | 5,087 | 4,456 (11.8%) | 3,685 |

The naive and impact-aware plans share only 5 of 200 identities: the objective
decides who gets fixed. The balanced plan keeps 68% / 73% of each.

### Research gaps

Source motivating the three gaps: Tenali & Potnuri, *A Comprehensive Review of
Recent Challenges and Emerging Trends in Malicious Insider Threat Detection
Using Machine Learning-Based Methods*, Springer LNNS vol. 2019, ICICC 2026
(multimodal fusion, graph-based modelling and explanation-fidelity metrics as
under-exploited trends).

**Gap 1 — Multi-signal risk fusion.** Logistic-regression meta-model over
XGBoost, Isolation Forest, temporal trend and blast radius; thresholds tuned on
train, evaluated on test.

| Signal | F1 | ROC-AUC |
|---|---|---|
| XGBoost | 0.882 | 0.987 |
| Isolation Forest | 0.691 | 0.895 |
| Temporal trend | 0.406 | 0.505 |
| Blast radius | 0.634 | 0.818 |
| **Fusion** | **0.883** | 0.980 |

F1 gain +0.0007, 95% bootstrap CI [−0.0067, +0.0081]: **not significant**.
XGBoost already captures what the other signals carry on this dataset.

**Gap 2 — SHAP explanation fidelity.** Deletion test on 2,000 test identities:
removing the top-5 SHAP features lowers the predicted-class probability by
0.238 vs 0.033 for 5 random features (**7.3x**). 60.5% of explanations are
high-fidelity (≥1.5x random) — **100% for predicted anomalies**, 47% for
predicted-normal identities (the model is so confident about them that
removing any five features barely matters). Sufficiency: the top-5 features
alone retain 0.959 of the prediction vs 0.774 for random five.
Explanation-action agreement: for 74.5% of identities the recommended action
changes one of their own top-5 SHAP risk factors (93% for *Reduce System
Footprint*, 0% for *Enforce MFA*, because explanations come from the anomaly
model and actions are scored with the risk-level model).

**Gap 3 — Graph-structural peer groups.** Cosine 10-NN graph over sensitive
resource access (9,771 identities, 81,355 edges), 51 Louvain communities
(modularity 0.557). 248 identities sit ≥2σ above their structural peers — all
true anomalies, but **0 novel** (all already flagged by Phase 1). Adding the
neighbours' risk does not raise test AUC (0.985 → 0.985).

## Limitations (reported, not hidden)

1. **Synthetic data.** Results show the pipeline works and is internally
   consistent; they are not evidence about real enterprise IAM data.
2. **Activity volume leaks risk.** The generator gives riskier identities more
   events, so `login_count` / `access_count` are the top SHAP features. A real
   deployment should normalise activity by role.
3. **No temporal escalation in the data.** Events are spread uniformly over the
   year, so trend detection cannot be demonstrated (AUC 0.495) even though the
   module works.
4. **Resources are accessed at random.** Graph peer groups therefore carry no
   extra risk signal (Gap 3), and fusion gains are not significant (Gap 1).
5. Counterfactual reductions are **model** predictions, not observed outcomes.
6. The compliance mapping is an assessment aid, not a certification.

Future work: inject role-based access patterns and time-escalating attack
campaigns into the generator, then re-run Gaps 1 and 3; class re-weighting for
rare threat types.
