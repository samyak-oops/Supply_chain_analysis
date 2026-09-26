"""
run_all.py – Master runner for all phases
Supply Chain Delay / Risk Analysis
"""
import subprocess, sys, os, time

BASE    = os.path.dirname(os.path.abspath(__file__))
NB_DIR  = os.path.join(BASE, "notebooks")

scripts = [
    ("01_data_cleaning.py",      "Phase 1 – Data Cleaning"),
    ("02_eda.py",                "Phase 2 – EDA"),
    ("03_root_cause_analysis.py","Phase 3 – Root Cause / Statistical Analysis"),
    ("04_predictive_model.py",   "Phase 4 – Predictive Model"),
    ("05_report_and_recs.py",    "Phase 5+6 – Recommendations & PDF Report"),
]

total_start = time.time()
for script, label in scripts:
    path = os.path.join(NB_DIR, script)
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, path], capture_output=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n❌ {label} FAILED (exit code {result.returncode})")
        sys.exit(1)
    print(f"\n✓ {label} completed in {elapsed:.1f}s")

print(f"\n{'='*60}")
print(f"ALL PHASES COMPLETE in {time.time()-total_start:.1f}s")
print(f"{'='*60}")
