#!/usr/bin/env python3
"""
X-UBA - MCA Major Project: run the complete pipeline with one command.

    python run_pipeline.py                 run all 14 modules in dependency order
    python run_pipeline.py --from 5        resume from step 5 (earlier outputs must exist)
    python run_pipeline.py --only 7        run a single step
    python run_pipeline.py --list          show the steps

Stops at the first failing module and exits with a non-zero code, so a
failure can never be reported as success. Works on Windows, Linux and macOS.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Phase 0  Dataset generation", "src/data_generation/generate_dataset.py"),
    ("Phase 1  XGBoost + SHAP", "src/modeling/train_models.py"),
    ("Phase 2  Isolation Forest", "src/anomaly_detection/train_isolation_forest.py"),
    ("Phase 3  K-Means clustering", "src/clustering/train_kmeans_clustering.py"),
    ("Phase 4  Temporal risk trajectory", "src/temporal_analysis/temporal_risk_trajectory.py"),
    ("Phase 5  Privilege graph + blast radius", "src/graph_analysis/build_privilege_graph.py"),
    ("Phase 6  Counterfactual engine", "src/simulator/counterfactual_engine.py"),
    ("Phase 7  Attack simulator", "src/simulator/attack_simulator.py"),
    ("Phase 8a SOD violations", "src/compliance/sod_violations.py"),
    ("Phase 8b Compliance gaps", "src/compliance/compliance_gap_analysis.py"),
    ("Phase 8c Organisational anomalies", "src/organizational_analysis/org_anomaly_detection.py"),
    ("Gap 1    Multi-signal risk fusion", "src/risk_fusion/risk_fusion_engine.py"),
    ("Gap 2    SHAP explanation fidelity", "src/explainability/explanation_fidelity.py"),
    ("Gap 3    Graph-structural anomalies", "src/graph_analysis/graph_structural_anomaly.py"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=1, help="first step number (1-14)")
    ap.add_argument("--only", type=int, help="run only this step number")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    ap.add_argument("--quiet", action="store_true", help="hide module output (show only status)")
    args = ap.parse_args()

    if args.list:
        for i, (name, path) in enumerate(STEPS, 1):
            print(f"{i:2d}. {name:42s} {path}")
        return 0

    if sys.version_info < (3, 10):
        print(f"[X] Python 3.10+ required, found {sys.version.split()[0]}")
        return 1

    selected = [args.only] if args.only else list(range(args.start, len(STEPS) + 1))
    if not all(1 <= s <= len(STEPS) for s in selected):
        print(f"[X] Step numbers must be between 1 and {len(STEPS)}")
        return 1

    print("=" * 70)
    print("X-UBA | MCA MAJOR PROJECT | FULL PIPELINE")
    print(f"Python {sys.version.split()[0]} | {len(selected)} step(s)")
    print("=" * 70)

    results = []
    t_all = time.time()
    for i in selected:
        name, path = STEPS[i - 1]
        print(f"\n>>> [{i}/{len(STEPS)}] {name}  ({path})", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(ROOT / path)], cwd=ROOT,
                              capture_output=args.quiet, text=True)
        dt = time.time() - t0
        results.append((i, name, proc.returncode == 0, dt))
        if proc.returncode != 0:
            if args.quiet:
                print(proc.stdout[-3000:])
                print(proc.stderr[-3000:])
            print(f"\n[X] {name} FAILED (exit code {proc.returncode}) - pipeline stopped.")
            break
        print(f"[OK] {name} finished in {dt:.1f}s", flush=True)

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    for i, name, ok, dt in results:
        print(f"  {'[OK]' if ok else '[X] '} {i:2d}. {name:42s} {dt:6.1f}s")
    all_ok = len(results) == len(selected) and all(ok for *_, ok, _ in results)
    print("-" * 70)
    print(f"  {'ALL STEPS PASSED' if all_ok else 'FAILED'} in {time.time() - t_all:.1f}s")
    if all_ok:
        print("  Outputs: data/raw, data/processed, models/, reports/")
        print("  Dashboard: python src/dashboard/app.py  ->  http://127.0.0.1:5000")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
