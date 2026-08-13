# Project Status — Final Reverified Build

| Component | Status | Verified implementation |
|---|---|---|
| Raw dataset | ✅ Complete | 586,737 raw interaction rows |
| Data validation/cleaning | ✅ Fixed | Required/null/category/type/flag/value validation + exact duplicate removal |
| Journey reconstruction | ✅ Fixed | Strictly pre-conversion impressions only; same-timestamp ambiguity excluded |
| First-touch attribution | ✅ Complete | Dynamically regenerated |
| Last-touch attribution | ✅ Complete | Dynamically regenerated |
| Markov attribution | ✅ Verified/documented | Absorbing transition matrix + explicit branch-to-Null removal counterfactual |
| Attribution comparison | ✅ Complete | Dynamic CSV/DB output |
| Budget simulation | ✅ Correctly scoped | Constant-spend attribution-aligned scenario; exact cent conservation |
| ML target definition | ✅ Fixed again | Fixed 7-day conversion horizon; late snapshots without complete follow-up excluded |
| Logistic Regression | ✅ Retrained | Validation-selected; untouched-test ROC-AUC 0.6474, PR-AUC 0.0643 |
| XGBoost | ✅ Retrained | Untouched-test ROC-AUC 0.6487, PR-AUC 0.0655 |
| ML feature engineering | ✅ Fixed | In-progress 1–5 touch prefixes only |
| Right-censoring | ✅ Fixed | 78,025 late-period prefix snapshots excluded from the 7-day target |
| Model-selection leakage | ✅ Fixed | Model selected on validation PR-AUC, never on test performance |
| Customer leakage | ✅ Fixed | Disjoint train/validation/test customer sets; overlap = 0 |
| Probability diagnostics | ✅ Verified | Brier/log loss/ECE/calibration gap reported on held-out data |
| Saved ML models | ✅ Complete | Full preprocessing + estimator pipelines serialized |
| Live prediction | ✅ Complete | API/dashboard load the selected serialized pipeline and return 7-day probability |
| Streamlit dashboard | ✅ Rebuilt | Dynamic artifacts and live model inference; no hardcoded results |
| FastAPI backend | ✅ Complete | Root, health, summary, attribution, budget, prediction endpoints |
| API readiness | ✅ Improved | Health checks database/model/attribution/metrics loadability and returns 503 when degraded |
| SQLite database | ✅ Complete | Integrity check = `ok`; five populated project tables |
| Automated testing | ✅ Expanded | 15 tests passing |
| Docker | ✅ Improved | Non-root image + API healthcheck + dashboard health dependency |
| GitHub CI | ✅ Improved | Compile → full rebuild → tests → Docker build → Compose API/Streamlit smoke test |
| Dependency reproducibility | ✅ Improved | Direct runtime dependencies pinned; Docker and CI both use Python 3.11 |
| Documentation | ✅ Updated | Current metrics, 7-day target, methodology, architecture, limitations, commands |
| Reproducibility | ✅ Verified | Generated data/attribution outputs reproduced from raw data; root-safe pipeline runner |
| Portfolio readiness | ✅ Ready | Production-style portfolio build with explicit methodological limits |

## Current verified outputs

- Raw rows: **586,737**
- Clean rows: **582,592**
- Modeled journeys: **232,648**
- Converted modeled journeys: **10,179**
- Prediction horizon: **7 days**
- Converted modeled journeys whose last eligible touch is within 7 days of conversion: **86.18%**
- Eligible ML snapshots: **393,810**
- Right-censored snapshots excluded: **78,025**
- Eligible ML customers: **199,997**
- Selected model: **Logistic Regression**
- Untouched-test ROC-AUC: **0.6474**
- Untouched-test PR-AUC: **0.0643**
- Brier score: **0.0350**
- 10-bin ECE: **0.0013**
- Customer overlap across train/validation/test: **0**
- Tests: **15 passed**
- SQLite integrity: **ok**
- FastAPI smoke endpoints: **passing**

## Important scope statement

The project is complete for its stated portfolio scope. It should **not** be described as causal attribution or a causal marketing-mix optimizer. The budget module demonstrates attribution-aligned reallocation under constant spend; causal budget optimization would require additional spend/response or experimental data.
