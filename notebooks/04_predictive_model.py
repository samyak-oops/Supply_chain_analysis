"""
Phase 4 - Predictive Modelling  (v2 - leakage-free features)
Supply Chain Delay / Risk Analysis

Features used: ONLY pre-shipment information available at order time:
  - Shipping Mode, Order Region, Market, Category Name  (categorical)
  - Order Item Quantity, Order Item Discount, Order Item Discount Rate  (numeric)
  - Days for Shipment Scheduled  (the PROMISED days - known at order time)

Excluded to prevent data leakage:
  - Days_for_shipping_real  (post-delivery)
  - delay_days              (derived from real vs scheduled - post-delivery)
  - Delivery_Status         (perfectly correlated with target)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (accuracy_score, precision_score, recall_score,
                                      f1_score, confusion_matrix, classification_report,
                                      roc_auc_score)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV  = os.path.join(BASE, "outputs", "cleaned_data.csv")
COL_MAP    = os.path.join(BASE, "outputs", "col_map.json")
CHARTS_DIR = os.path.join(BASE, "outputs", "charts")
MODEL_DIR  = os.path.join(BASE, "outputs", "models")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BG_COLOR = "#F7F9FC"
PALETTE  = "#E84855"
SAFE_COL = "#3A86FF"

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading cleaned data ...")
df = pd.read_csv(CLEAN_CSV)
with open(COL_MAP) as f:
    cm = json.load(f)

TARGET     = cm.get("late_risk")
SHIP_MODE  = cm.get("shipping_mode")
REGION     = cm.get("region")
MARKET     = cm.get("market")
CATEGORY   = cm.get("category")
QTY        = cm.get("quantity")
DISCOUNT   = cm.get("discount")
DAYS_SCHED = cm.get("days_scheduled")

# Ensure target is numeric
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)

# ── Feature engineering ────────────────────────────────────────────────────────
# ONLY pre-shipment columns; exclude leakage sources explicitly
LEAKAGE_COLS = {
    cm.get("days_real"),       # Days_for_shipping_real - post-delivery
    cm.get("delay_days"),      # delay_days - derived from real - post-delivery
    cm.get("delivery_status"), # Delivery_Status - perfectly correlated with target
    cm.get("shipping_date"),   # actual ship date - post-delivery
    TARGET
}
LEAKAGE_COLS = {c for c in LEAKAGE_COLS if c}

cat_cols = [c for c in [SHIP_MODE, REGION, MARKET, CATEGORY] if c and c in df.columns]
num_cols = [c for c in [QTY, DISCOUNT, DAYS_SCHED] if c and c in df.columns]

# Also include Order_Item_Discount_Rate if present
discount_rate_col = None
for c in df.columns:
    if "discount_rate" in c.lower():
        discount_rate_col = c
        break
if discount_rate_col and discount_rate_col not in num_cols:
    num_cols.append(discount_rate_col)

print(f"Categorical features: {cat_cols}")
print(f"Numeric features    : {num_cols}")
print(f"Leakage cols excluded: {LEAKAGE_COLS}")

# One-hot encode categoricals
encode_df = pd.get_dummies(df[cat_cols + num_cols + [TARGET]], columns=cat_cols, drop_first=False)
X = encode_df.drop(columns=[TARGET])
y = encode_df[TARGET]
X = X.fillna(X.median(numeric_only=True))

# Convert bool columns (from get_dummies) to int so they pass select_dtypes
bool_cols = X.select_dtypes(include="bool").columns.tolist()
if bool_cols:
    X[bool_cols] = X[bool_cols].astype(int)

X = X.select_dtypes(include=[np.number])
feature_names = X.columns.tolist()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution (%):\n{y.value_counts(normalize=True).mul(100).round(1)}")

# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

metrics_all = {}

def evaluate(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_prob) if y_prob is not None else None
    cm_  = confusion_matrix(y_te, y_pred)

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if auc: print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_te, y_pred)}")
    print(f"\n  Confusion Matrix:\n{cm_}")

    # Confusion matrix chart
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG_COLOR)
    labels_cm = ["On-Time", "Late"]
    sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels_cm, yticklabels=labels_cm,
                cbar=False, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Confusion Matrix - {name}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    cm_path = os.path.join(CHARTS_DIR, f"cm_{safe_name}.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Confusion matrix saved: cm_{safe_name}.png")

    metrics_all[name] = {
        "accuracy":         round(acc, 4),
        "precision":        round(prec, 4),
        "recall":           round(rec, 4),
        "f1":               round(f1, 4),
        "roc_auc":          round(auc, 4) if auc else None,
        "confusion_matrix": cm_.tolist()
    }
    return y_pred

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1: Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING: Logistic Regression")
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced",
                              solver="lbfgs"))
])
lr_pipe.fit(X_train, y_train)
evaluate("Logistic Regression", lr_pipe, X_test, y_test)

# Coefficients
lr_coefs = pd.Series(
    lr_pipe.named_steps["lr"].coef_[0],
    index=feature_names
).sort_values(key=abs, ascending=False).head(20)

print(f"\nTop 20 LR Coefficients:\n{lr_coefs.round(4)}")

fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
bar_colors = [PALETTE if c > 0 else SAFE_COL for c in lr_coefs.values]
ax.barh(lr_coefs.index[::-1], lr_coefs.values[::-1], color=bar_colors[::-1], edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient Value", fontsize=10)
ax.set_title("Logistic Regression - Top Feature Coefficients\n(Red = increases late risk, Blue = decreases)",
             fontsize=10, fontweight="bold")
ax.set_facecolor(BG_COLOR)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
lr_path = os.path.join(CHARTS_DIR, "08_lr_coefficients.png")
fig.savefig(lr_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close(fig)
print("LR coefficients chart saved.")
joblib.dump(lr_pipe, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2: Random Forest
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING: Random Forest (200 trees, max_depth=12) ...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
evaluate("Random Forest", rf, X_test, y_test)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False).head(20)
print(f"\nTop 20 RF Feature Importances:\n{importances.round(4)}")

fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
imp_colors = [PALETTE if i < 5 else "#6C757D" for i in range(len(importances))]
ax.barh(importances.index[::-1], importances.values[::-1],
        color=imp_colors[::-1], edgecolor="white")
ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=10)
ax.set_title("Random Forest - Top 20 Feature Importances", fontsize=11, fontweight="bold")
ax.set_facecolor(BG_COLOR)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fi_path = os.path.join(CHARTS_DIR, "09_rf_feature_importance.png")
fig.savefig(fi_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close(fig)
print("Feature importance chart saved.")
joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3: XGBoost (optional)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from xgboost import XGBClassifier
    print("\n" + "="*60)
    print("TRAINING: XGBoost ...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    evaluate("XGBoost", xgb, X_test, y_test)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgboost.pkl"))
    print("XGBoost model saved.")
except ImportError:
    print("\nXGBoost not installed - skipping.")

# ── Model comparison chart ────────────────────────────────────────────────────
metric_df = pd.DataFrame(metrics_all).T[["accuracy","precision","recall","f1"]].reset_index()
metric_df.rename(columns={"index":"Model"}, inplace=True)

fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG_COLOR)
palette_map = {"accuracy": "#3A86FF", "precision": "#FFBE0B",
               "recall": PALETTE, "f1": "#8338EC"}
x = np.arange(len(metric_df))
bar_w = 0.2
metrics_list = ["accuracy","precision","recall","f1"]
for i, m in enumerate(metrics_list):
    vals = metric_df[m].astype(float)
    bars = ax.bar(x + i * bar_w, vals, bar_w, label=m.capitalize(),
                  color=palette_map[m], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + bar_w * 1.5)
ax.set_xticklabels(metric_df["Model"], fontsize=9)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.12)
ax.set_title("Model Comparison - Accuracy / Precision / Recall / F1",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_facecolor(BG_COLOR)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
comp_path = os.path.join(CHARTS_DIR, "10_model_comparison.png")
fig.savefig(comp_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close(fig)
print("Model comparison chart saved.")

# ── Save metrics ──────────────────────────────────────────────────────────────
metrics_path = os.path.join(BASE, "outputs", "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_all, f, indent=2)
print(f"\nMetrics saved -> {metrics_path}")

# ── Why recall > accuracy ─────────────────────────────────────────────────────
print("""
WHY RECALL > ACCURACY IN THIS CONTEXT
--------------------------------------
In supply chain delay prediction, a False Negative (predicting on-time when
the shipment is actually late) is far more costly than a False Positive:

  - False Negative (missed late) -> customer dissatisfaction, SLA breach,
    expediting costs, and reputational damage.
  - False Positive (false alarm)  -> minor over-allocation of expediting
    resources (manageable and reversible).

With a 55/45 class split, accuracy can be misleading; a naive model that
always predicts 'late' would achieve 54.8% accuracy. We therefore optimise
for RECALL (catch as many true lates as possible) to minimise operational harm.
""")

print("\nPHASE 4 COMPLETE.")
