# Resume Bullets – Supply Chain Delay & Risk Analysis

---

## ONE-LINE BULLET

**Supply Chain Delay & Risk Analysis** — Analysed 180,519 orders using Python (pandas, scikit-learn, XGBoost); identified First Class shipping (95.3% late rate, Cramér's V=0.457) as the dominant delay driver and built a predictive model achieving 54.8% recall and 0.73 ROC-AUC on a leakage-free 86-feature set, with findings packaged into an executive PDF report and actionable carrier-renegotiation recommendations.

---

## THREE-LINE BULLET

**Supply Chain Delay & Risk Analysis** | Python, pandas, scikit-learn, XGBoost, ReportLab
- Cleaned and analysed a 180,519-row global supply chain dataset (DataCo); surfaced that **54.8% of all orders are late**, with First Class shipping exhibiting a **95.3% late-delivery rate** — 40.5 percentage points above the fleet average — confirmed statistically significant via chi-square test (χ² p<0.001, Cramér's V=0.457).
- Engineered 86 pre-shipment features (one-hot encoded shipping mode, region, category; order quantity, discount, scheduled ship days) and trained Logistic Regression, Random Forest (200 trees), and XGBoost classifiers; best model achieved **69.6% accuracy, 84.3% precision, 54.8% recall, and 0.73 ROC-AUC** — optimising for recall to minimise SLA-breach False Negatives.
- Delivered a professional PDF report (ReportLab) embedding 10 visualisations, statistical root-cause tables, model comparison charts, and 5 numbers-backed business recommendations — including a projected **12 pp reduction in overall late rate** via carrier-contract restructuring for First Class shipments.

---

## CONTEXT NOTE

All numbers above are computed from the actual DataCo dataset:
- Dataset: 180,519 rows
- Overall late delivery rate: 54.8% (98,977 orders)
- First Class late rate: 95.3% | Second Class: 76.6% | Same Day: 45.7% | Standard Class: 38.1%
- Cramér's V (Shipping Mode vs Late Risk): 0.457 (large effect size)
- Best model recall: 54.8% (XGBoost) | F1: 66.4% | ROC-AUC: 0.729
- Features: 86 (4 numeric + 82 one-hot encoded from 4 categorical columns)
- Estimated impact of restructuring First Class: ~12 pp reduction in overall late rate
