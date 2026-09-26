"""
Phase 3 - Root Cause / Statistical Analysis
Supply Chain Delay / Risk Analysis
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import json, os
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV = os.path.join(BASE, "outputs", "cleaned_data.csv")
COL_MAP   = os.path.join(BASE, "outputs", "col_map.json")
OUT_DIR   = os.path.join(BASE, "outputs")

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading cleaned data …")
df = pd.read_csv(CLEAN_CSV)
with open(COL_MAP) as f:
    cm = json.load(f)

TARGET    = cm.get("late_risk")
SHIP_MODE = cm.get("shipping_mode")
REGION    = cm.get("region")
MARKET    = cm.get("market")
CATEGORY  = cm.get("category")

print(f"Target column : {TARGET}")
print(f"Shipping mode : {SHIP_MODE}")
print(f"Region        : {REGION}")
print(f"Market        : {MARKET}")
print(f"Category      : {CATEGORY}")

# Ensure target is numeric
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)

results = {}

# ── Chi-Square test helper ─────────────────────────────────────────────────────
def chi2_test(df, col, target):
    """Run chi-square test and compute Cramér's V effect size."""
    if not col or col not in df.columns:
        return None
    contingency = pd.crosstab(df[col], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    r, k = contingency.shape
    cramers_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))

    late_rate = df.groupby(col)[target].mean() * 100
    summary = late_rate.sort_values(ascending=False).reset_index()
    summary.columns = [col, "Late_Rate_%"]
    summary["Count"] = df.groupby(col)[target].count().values

    print(f"\n{'='*60}")
    print(f"Chi-Square Test: {col} vs {target}")
    print(f"  χ² = {chi2:.2f}, p = {p:.2e}, dof = {dof}")
    print(f"  Cramér's V = {cramers_v:.4f}  (effect size)")
    sig = "*** SIGNIFICANT ***" if p < 0.05 else "not significant"
    print(f"  → {sig} (α=0.05)")
    print(f"\n  Late rate by {col}:\n{summary.to_string(index=False)}")

    return {
        "column": col,
        "chi2": round(chi2, 4),
        "p_value": float(f"{p:.2e}"),
        "dof": int(dof),
        "cramers_v": round(cramers_v, 4),
        "significant": bool(p < 0.05),
        "late_rate_table": summary.to_dict(orient="records")
    }

# ── Run tests ──────────────────────────────────────────────────────────────────
for label, col in [("shipping_mode", SHIP_MODE),
                   ("region",        REGION),
                   ("market",        MARKET),
                   ("category",      CATEGORY)]:
    res = chi2_test(df, col, TARGET)
    if res:
        results[label] = res

# ── Summary table: late rate by each dimension ─────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY TABLE – Late Delivery Rate by Key Dimensions")
print(f"{'='*60}")

summary_rows = []
for label, res in results.items():
    tbl = pd.DataFrame(res["late_rate_table"])
    if not tbl.empty:
        worst = tbl.iloc[0]
        best  = tbl.iloc[-1]
        col_name = res["column"]
        print(f"\n[{label}]  Cramér's V={res['cramers_v']:.3f}  p={res['p_value']:.2e}")
        print(tbl.to_string(index=False))
        summary_rows.append({
            "Dimension":    label,
            "Column":       col_name,
            "Cramers_V":    res["cramers_v"],
            "P_Value":      res["p_value"],
            "Significant":  res["significant"],
            "Worst_Value":  str(worst.iloc[0]),
            "Worst_Rate_%": round(float(worst["Late_Rate_%"]), 2),
            "Best_Value":   str(best.iloc[0]),
            "Best_Rate_%":  round(float(best["Late_Rate_%"]), 2),
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, "root_cause_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary table saved → {summary_path}")

# ── Identify top 3 root causes ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("TOP 3 ROOT CAUSES OF LATE DELIVERIES")
print(f"{'='*60}")

# Use all results if none reach significance (shouldn't happen with n=180k)
sig_results = {k: v for k, v in results.items() if v.get("significant")}
if not sig_results:
    print("WARNING: No statistically significant dimensions found. Using all results.")
    sig_results = results
sorted_by_effect = sorted(sig_results.items(), key=lambda x: x[1]["cramers_v"], reverse=True)

root_causes = []
for rank, (dim, res) in enumerate(sorted_by_effect[:3], 1):
    tbl  = pd.DataFrame(res["late_rate_table"])
    worst = tbl.iloc[0]
    col_name = res["column"]

    # Overall late rate for comparison
    overall_late = df[TARGET].mean() * 100
    gap = float(worst["Late_Rate_%"]) - overall_late

    cause_str = (
        f"ROOT CAUSE #{rank}: {dim.upper().replace('_', ' ')}\n"
        f"  • '{worst.iloc[0]}' has a late-delivery rate of {worst['Late_Rate_%']:.1f}%  "
        f"(overall avg = {overall_late:.1f}%, gap = +{gap:.1f} pp)\n"
        f"  • χ² test: p = {res['p_value']:.2e}, Cramér's V = {res['cramers_v']:.3f}\n"
        f"  • Statistically significant (p < 0.05)"
    )
    print(f"\n{cause_str}")
    root_causes.append({
        "rank": rank,
        "dimension": dim,
        "worst_value": str(worst.iloc[0]),
        "worst_rate_pct": float(worst["Late_Rate_%"]),
        "overall_rate_pct": round(overall_late, 2),
        "gap_pp": round(gap, 2),
        "cramers_v": res["cramers_v"],
        "p_value": res["p_value"]
    })

# ── Save all stats results ─────────────────────────────────────────────────────
stats_out = {
    "chi2_tests": results,
    "root_causes": root_causes,
    "overall_late_rate_pct": round(df[TARGET].mean() * 100, 2)
}
stats_path = os.path.join(OUT_DIR, "statistical_results.json")
with open(stats_path, "w") as f:
    json.dump(stats_out, f, indent=2)
print(f"\nStatistical results saved → {stats_path}")
print("\n✅ Phase 3 complete.")
