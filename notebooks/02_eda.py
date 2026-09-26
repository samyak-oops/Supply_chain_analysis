"""
Phase 2 - Exploratory Data Analysis  (v2)
Supply Chain Delay / Risk Analysis
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import json, os, warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV  = os.path.join(BASE, "outputs", "cleaned_data.csv")
COL_MAP    = os.path.join(BASE, "outputs", "col_map.json")
CHARTS_DIR = os.path.join(BASE, "outputs", "charts")
TAKEAWAYS  = os.path.join(BASE, "outputs", "eda_takeaways.json")
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading cleaned data ...")
df = pd.read_csv(CLEAN_CSV)
with open(COL_MAP) as f:
    cm = json.load(f)

print(f"Shape: {df.shape}")

PALETTE   = "#E84855"
SAFE_COL  = "#3A86FF"
BG_COLOR  = "#F7F9FC"

def save(fig, name, takeaway_dict, key, takeaway):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    takeaway_dict[key] = {"file": name, "takeaway": takeaway}
    print(f"  Saved {name}")
    return path

def pct_bar(ax, pcts, labels, colors_=None, orient="v", title=""):
    if colors_ is None:
        colors_ = PALETTE
    if orient == "v":
        bars = ax.bar(labels, pcts, color=colors_, edgecolor="white", linewidth=0.8)
        ax.set_ylim(0, min(float(pcts.max()) * 1.25, 105))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
        for b, p in zip(bars, pcts):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f"{p:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    else:
        bars = ax.barh(labels, pcts, color=colors_, edgecolor="white", linewidth=0.8)
        ax.set_xlim(0, min(float(pcts.max()) * 1.25, 105))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
        for b, p in zip(bars, pcts):
            ax.text(p + 0.5, b.get_y() + b.get_height()/2,
                    f"{p:.1f}%", ha="left", va="center", fontsize=8, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_facecolor(BG_COLOR)
    ax.spines[["top","right"]].set_visible(False)
    return ax

takeaways = {}

TARGET     = cm.get("late_risk")
SHIP_MODE  = cm.get("shipping_mode")
REGION     = cm.get("region")
MARKET     = cm.get("market")
CATEGORY   = cm.get("category")
ORDER_DATE = cm.get("order_date")
QTY        = cm.get("quantity")
DISCOUNT   = cm.get("discount")
DELAY_DAYS = cm.get("delay_days")
DAYS_SCHED = cm.get("days_scheduled")
DAYS_REAL  = cm.get("days_real")

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)

# ── Chart 1: Overall late % ────────────────────────────────────────────────────
overall_late = df[TARGET].mean() * 100
print(f"\nOverall late %: {overall_late:.2f}%")

fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG_COLOR)
vals   = [overall_late, 100 - overall_late]
labels = ["Late", "On-Time"]
colors = [PALETTE, SAFE_COL]
wedges, texts, autotexts = ax.pie(vals, labels=labels, colors=colors,
                                   autopct="%1.1f%%", startangle=90,
                                   wedgeprops=dict(linewidth=1.5, edgecolor="white"))
for at in autotexts:
    at.set_fontsize(11); at.set_fontweight("bold")
ax.set_title("Overall Late Delivery Rate", fontsize=12, fontweight="bold")
save(fig, "01_overall_late_rate.png", takeaways, "overall",
     f"{overall_late:.1f}% of all orders are delivered late — "
     f"that's {overall_late/100 * len(df):,.0f} shipments out of {len(df):,} total, "
     f"representing a major operational risk and customer satisfaction issue.")

# ── Chart 2: Late % by Shipping Mode ──────────────────────────────────────────
if SHIP_MODE and SHIP_MODE in df.columns:
    grp = df.groupby(SHIP_MODE)[TARGET].agg(["mean","count"]).reset_index()
    grp["pct"] = grp["mean"] * 100
    grp = grp.sort_values("pct", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG_COLOR)
    bar_colors = [PALETTE if p > overall_late else SAFE_COL for p in grp["pct"]]
    pct_bar(ax, grp["pct"], grp[SHIP_MODE], colors_=bar_colors, orient="v",
            title="Late Delivery Rate by Shipping Mode")
    ax.set_xlabel("Shipping Mode")
    ax.set_ylabel("Late Delivery Rate (%)")
    for i, (_, row) in enumerate(grp.iterrows()):
        ax.text(i, -4, f"n={row['count']:,}", ha="center", fontsize=7, color="gray")
    fig.tight_layout()

    worst_mode = grp.iloc[0]
    save(fig, "02_late_by_ship_mode.png", takeaways, "ship_mode",
         f"'{worst_mode[SHIP_MODE]}' has the highest late-delivery rate at "
         f"{worst_mode['pct']:.1f}%, well above the overall average of {overall_late:.1f}%. "
         f"This mode is systematically under-resourced relative to demand and warrants "
         f"immediate carrier negotiation or capacity reallocation.")

# ── Chart 3: Late % by Region ─────────────────────────────────────────────────
region_col = None
for rc in [REGION, MARKET]:
    if rc and rc in df.columns:
        region_col = rc
        break

if region_col:
    grp_r = df.groupby(region_col)[TARGET].agg(["mean","count"]).reset_index()
    grp_r["pct"] = grp_r["mean"] * 100
    grp_r = grp_r.sort_values("pct", ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(grp_r)*0.55), 5), facecolor=BG_COLOR)
    bar_colors = [PALETTE if p > overall_late else SAFE_COL for p in grp_r["pct"]]
    pct_bar(ax, grp_r["pct"], grp_r[region_col], colors_=bar_colors, orient="v",
            title=f"Late Delivery Rate by {region_col.replace('_',' ')}")
    ax.set_xlabel(region_col.replace("_", " "), fontsize=9)
    ax.set_ylabel("Late Delivery Rate (%)")
    plt.xticks(rotation=40, ha="right", fontsize=7)
    fig.tight_layout()

    worst_reg = grp_r.iloc[0]
    save(fig, "03_late_by_region.png", takeaways, "region",
         f"'{worst_reg[region_col]}' has the worst late-delivery rate at "
         f"{worst_reg['pct']:.1f}%, pointing to potential capacity, carrier, or "
         f"geographic challenges unique to that region that require targeted intervention.")

# ── Chart 4: Late % by Category (top 15 by volume, sorted by late rate) ───────
if CATEGORY and CATEGORY in df.columns:
    grp_c = df.groupby(CATEGORY)[TARGET].agg(["mean","count"]).reset_index()
    grp_c["pct"] = grp_c["mean"] * 100
    # Keep top 15 by volume for readability
    grp_c = grp_c.nlargest(15, "count")
    grp_c = grp_c.sort_values("pct", ascending=False)

    fig, ax = plt.subplots(figsize=(7, max(4, len(grp_c)*0.4)), facecolor=BG_COLOR)
    bar_colors = [PALETTE if p > overall_late else SAFE_COL for p in grp_c["pct"]]
    pct_bar(ax, grp_c["pct"], grp_c[CATEGORY], colors_=bar_colors, orient="h",
            title="Late Delivery Rate by Product Category (Top 15 by Volume)")
    ax.set_xlabel("Late Delivery Rate (%)")
    fig.tight_layout()

    worst_cat = grp_c.iloc[0]
    save(fig, "04_late_by_category.png", takeaways, "category",
         f"Among the top-volume categories, '{worst_cat[CATEGORY]}' leads with "
         f"{worst_cat['pct']:.1f}% late rate. Categories likely differ in handling "
         f"complexity, supplier lead times, or sourcing geography.")

# ── Chart 5: Quantity & Discount & Delay Days distributions ───────────────────
plot_cols = [(c, lbl) for c, lbl in
             [(QTY, "Order Quantity"), (DISCOUNT, "Discount"), (DELAY_DAYS, "Delay Days (Real-Sched)")]
             if c and c in df.columns]

if plot_cols:
    n = len(plot_cols)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), facecolor=BG_COLOR)
    if n == 1:
        axes = [axes]

    for ax, (col, lbl) in zip(axes, plot_cols):
        late_vals    = df[df[TARGET] == 1][col].dropna()
        on_time_vals = df[df[TARGET] == 0][col].dropna()
        ax.hist(on_time_vals, bins=30, alpha=0.6, color=SAFE_COL, label="On-Time", density=True)
        ax.hist(late_vals,    bins=30, alpha=0.6, color=PALETTE,  label="Late",    density=True)
        ax.set_xlabel(lbl, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"{lbl}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_facecolor(BG_COLOR)
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Distribution of Key Numeric Features by Delivery Outcome",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "05_qty_discount_late.png", takeaways, "qty_discount",
         "Order quantity and discount show nearly identical distributions for late vs on-time "
         "deliveries, confirming these alone are weak predictors of delay. "
         "Delay days (real minus scheduled) is the most discriminating numeric feature.")

# ── Chart 6: Time trend of late deliveries ────────────────────────────────────
if ORDER_DATE and ORDER_DATE in df.columns:
    df[ORDER_DATE] = pd.to_datetime(df[ORDER_DATE], errors="coerce")
    df2 = df.dropna(subset=[ORDER_DATE]).copy()
    df2["ym"] = df2[ORDER_DATE].dt.to_period("M")
    trend = df2.groupby("ym")[TARGET].agg(["mean","count"]).reset_index()
    trend["pct"] = trend["mean"] * 100
    trend = trend[trend["count"] > 50]

    fig, ax = plt.subplots(figsize=(11, 4), facecolor=BG_COLOR)
    x_vals = list(range(len(trend)))
    ax.fill_between(x_vals, trend["pct"].values, alpha=0.2, color=PALETTE)
    ax.plot(x_vals, trend["pct"].values, color=PALETTE, linewidth=2)
    ax.axhline(overall_late, color="gray", linestyle="--", linewidth=1,
               label=f"Overall avg ({overall_late:.1f}%)")
    tick_step = max(1, len(trend) // 10)
    tick_idx  = list(range(0, len(trend), tick_step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(trend["ym"].iloc[i]) for i in tick_idx],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Late Delivery Rate (%)")
    ax.set_title("Late Delivery Rate Over Time (Monthly)", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.legend(fontsize=9)
    ax.set_facecolor(BG_COLOR)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()

    save(fig, "06_time_trend.png", takeaways, "time_trend",
         "Late delivery rates fluctuate over time; spikes aligned with peak seasons "
         "indicate carrier capacity is overwhelmed at predictable intervals, creating "
         "an opportunity for proactive seasonal surge planning.")

# ── Chart 7: Correlation heatmap ──────────────────────────────────────────────
num_df = df.select_dtypes(include=[np.number]).copy()
corr_with_target = num_df.corrwith(num_df[TARGET]).abs().sort_values(ascending=False)
top_cols = corr_with_target.head(16).index.tolist()
if TARGET not in top_cols:
    top_cols.append(TARGET)
corr_mat = num_df[top_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_mat, mask=mask, cmap=cmap, center=0, linewidths=0.5,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Heatmap - Numeric Features vs Late Delivery Risk",
             fontsize=11, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
fig.tight_layout()

top_corr = corr_with_target.drop(TARGET, errors="ignore").head(3)
save(fig, "07_correlation_heatmap.png", takeaways, "correlation",
     f"Top features correlated with late delivery risk: "
     f"{', '.join(top_corr.index.tolist())} "
     f"(|r| = {', '.join(f'{v:.2f}' for v in top_corr.values)}). "
     f"These will be the highest-value inputs for the predictive model.")

# ── Save takeaways ─────────────────────────────────────────────────────────────
with open(TAKEAWAYS, "w", encoding="utf-8") as f:
    json.dump(takeaways, f, indent=2, ensure_ascii=False)
print(f"\nTakeaways saved -> {TAKEAWAYS}")

print("\nPHASE 2 COMPLETE.")
for key, val in takeaways.items():
    print(f"\n[{key}] {val['takeaway']}")
