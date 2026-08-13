# Final Reverification Report

## Scope of this pass

The uploaded "FINAL_VERIFIED" build was unpacked and re-audited independently rather than trusted from its existing test suite. The review covered source code, raw-data semantics, journey leakage controls, Markov attribution, budget conservation, model target definition, train/validation/test design, probability diagnostics, serialized-model inference, FastAPI behavior, SQLite integrity, dependency reproducibility, Docker/Compose configuration, CI behavior, documentation claims, and regeneration from the supplied raw CSV.

## Additional issue found in the uploaded build

### Inconsistent future follow-up in the ML target

The previous model labeled a prefix positive when the customer converted at any later point before the dataset ended. A snapshot near July 1 therefore had almost a month in which to become positive, while a snapshot near July 31 had only minutes or hours. This is a right-censoring problem: the displayed "conversion probability" did not correspond to one consistent prediction horizon.

### Fix applied

The ML task is now explicitly **conversion within 7 days after the observed 1–5-touch prefix**.

- Dataset observation end: **2018-07-31 23:59:30 UTC**
- Latest eligible snapshot: **2018-07-24 23:59:30 UTC**
- Eligible prefix snapshots: **393,810**
- Late/right-censored prefix snapshots excluded: **78,025**
- Eligible customers: **199,997**
- **86.18%** of converted modeled journeys have their last eligible impression within 7 days of conversion, which supports 7 days as a practical horizon while retaining roughly three weeks of fully observed snapshot dates.

The customer split now operates only on customers with eligible fixed-horizon snapshots and stratifies using the fixed-horizon target. Customers remain disjoint across train, validation, and untouched test sets.

## Other hardening in this pass

- API `/health` now requires metrics artifacts as well as database/model/attribution readiness and returns HTTP 503 when degraded, so Docker health status cannot falsely pass on an unready service.
- `/project-summary`, `/attribution`, `/budget`, and `/predict` return service-unavailable responses when required generated artifacts cannot be loaded rather than silently presenting incomplete results.
- `/predict` explicitly returns `prediction_horizon_days = 7` and a fixed-horizon scope description.
- Streamlit now labels the output as **7-day conversion probability** and explains complete-horizon eligibility.
- Direct dependencies are pinned for a reproducible Python 3.11 Docker/CI runtime.
- CI now performs a Docker Compose smoke test after image build, including API health, attribution, prediction, and the Streamlit `/_stcore/health` endpoint.
- Automated tests expanded from 12 to **15**, including fixed-horizon positive labeling, late-snapshot censoring, API horizon metadata, and invalid API inputs.

## Reverified data and attribution outputs

- Raw rows: **586,737**
- Duplicate rows removed: **4,145**
- Clean rows: **582,592**
- Modeled journeys: **232,648**
- Converted modeled journeys: **10,179**
- Same-timestamp impressions excluded: **194**
- Baseline Markov conversion probability: **0.043753**
- $100,000 budget simulation conserves total spend exactly to the cent.

Markov shares remain:

- Facebook: **30.45%**
- Instagram: **17.76%**
- Online Display: **10.71%**
- Online Video: **18.84%**
- Paid Search: **22.24%**

The Markov module is intentionally described as an **observational first-order removal-effect model with an explicit branch-to-Null counterfactual**. It is not claimed to prove causal incrementality or to be identical to every Markov attribution package's removal operator.

## Reverified ML results

| Model | Validation PR-AUC | Untouched Test ROC-AUC | Untouched Test PR-AUC | Brier | 10-bin ECE |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **0.0690** | 0.6474 | 0.0643 | 0.0350 | 0.0013 |
| XGBoost | 0.0687 | 0.6487 | 0.0655 | 0.0350 | 0.0015 |

**Selected model: Logistic Regression**, based only on validation PR-AUC. The project does not change the selected model after observing the untouched test results.

Selected-model untouched-test diagnostics:

- observed 7-day positive rate: **3.6733%**
- mean predicted probability: **3.7465%**
- calibration gap: **+0.0732 percentage points**
- Brier score: **0.0350**
- 10-bin ECE: **0.13 percentage points**
- customer overlap across train/validation/test: **0**

## Runtime verification

- Python compilation: **passed**
- Pytest: **15 passed**
- FastAPI `/`: **200**
- FastAPI `/health`: **200 / ready = true**
- FastAPI `/project-summary`: **200**
- FastAPI `/attribution`: **200**
- FastAPI `/budget`: **200**
- FastAPI `/predict`: **200**
- Invalid prediction inputs: **422**
- Invalid zero budget: **422**
- SQLite `PRAGMA integrity_check`: **ok**
- SQLite journey rows: **232,648**
- Attribution rows: **5**
- Budget rows: **5**
- Model artifact loads and predicts: **passed**

## Environment limitation

This verification container has no Docker executable and does not have Streamlit installed. An attempted package install could not reach PyPI because outbound package-network access is disabled in the execution environment. Therefore a local interactive Streamlit process and Docker daemon could not be launched here. The dashboard source compiles, API/model/data paths are tested, Compose configuration is included, and GitHub CI is configured to install the pinned dependencies, launch the Compose stack, and smoke-test both service health endpoints.

## Final assessment

No remaining code defect was found in the tested core pipeline after the fixed-horizon correction. The implementation is **portfolio-ready and methodologically defensible for its stated scope**. "Perfect" should not be interpreted as removing limitations imposed by the dataset: attribution remains observational, the source covers one month, the propensity model is path-only, and causal ROI optimization cannot be established without spend-response or experimental data.
