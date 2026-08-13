# Marketing Attribution & Budget Intelligence Platform

A reproducible **full-stack ML analytics project** that reconstructs customer journeys, compares multi-touch attribution methods, simulates fixed-spend budget reallocation, and estimates conversion propensity from in-progress ad-impression paths.

## Business problem

Customers often see several ads before converting. Last-touch attribution gives all credit to the final observed marketing touch and ignores the earlier sequence. This project compares **First Touch**, **Last Touch**, and a first-order **Markov removal-effect model** to show how channel credit changes when the full observed path is considered.

The project then uses the Markov attribution shares in a **constant-total-spend reallocation simulation**. This is deliberately not presented as causal ROI optimization because the source data do not contain historical channel spend, response curves, randomized experiments, or incrementality measurements.

A separate ML module answers a different question: **given the first 1–5 ad impressions observed so far, what is the estimated probability that the customer converts within the next 7 days?** The target uses only snapshots with a complete 7-day follow-up window, so late-month right-censored examples are excluded.

---

## Verified dataset summary

| Metric | Verified value |
|---|---:|
| Raw interaction rows | 586,737 |
| Exact duplicate rows removed | 4,145 |
| Clean interaction rows | 582,592 |
| Unique customers | 240,108 |
| Conversion events | 17,639 |
| Modeled journeys with a strictly prior impression | 232,648 |
| Converted modeled journeys | 10,179 |
| Conversions without a strictly prior impression | 7,460 |
| Same-timestamp impressions excluded | 194 |
| Post-conversion impressions found/excluded | 0 |
| Channels | 5 |

Data window: **2018-07-01 to 2018-07-31 UTC**.

### Why the strict timestamp rule matters

For converting customers, only impressions **strictly earlier** than the first conversion timestamp enter the modeled journey. Impressions recorded at exactly the same timestamp as conversion are excluded because the source data do not provide a reliable within-second event order. This prevents outcome-time information from entering attribution or ML features.

---

## Architecture

```text
                           attribution_data.csv
                                   |
                                   v
                         Validation + cleaning
                                   |
                                   v
                     Leakage-safe journey builder
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
        First / Last / Markov attribution     Prefix snapshots
                    |                             |
                    v                       Train / Validation / Test
          Fixed-spend budget simulation          |
                    |                       LR + XGBoost
                    |                             |
                    +--------------+--------------+
                                   |
                         Generated artifacts
                                   |
                              SQLite DB
                                   |
                              FastAPI
                    +--------------+--------------+
                    |                             |
                    v                             v
             Streamlit dashboard            External clients
```

---

## Methodology

### 1. Data validation and journey reconstruction

The loader validates:

- required columns and non-null required values;
- non-empty customer identifiers;
- valid timestamps;
- expected channel and interaction categories;
- binary conversion flags;
- consistency between `interaction` and `conversion`;
- numeric, non-negative conversion values.

Exact duplicate rows are removed. Events are sorted by customer and timestamp. For each converting customer, the path ends before the first conversion event.

### 2. Attribution

**First Touch** gives full conversion credit to the first eligible impression channel.

**Last Touch** gives full conversion credit to the last eligible impression channel.

**Markov attribution** builds an absorbing first-order transition system:

```text
Start -> channel states -> Conversion / Null
```

The baseline conversion probability is computed from the transition matrix. For each channel, the project applies an explicit removal counterfactual: probability mass that would enter the unavailable channel is redirected to `Null`, after which the absorption probability of `Conversion` is recomputed.

```text
Removal Effect = 1 - P(Conversion | channel unavailable) / P(Conversion)
```

Positive removal effects are normalized into channel shares that sum to 100%.

> Important: this is an observational Markov removal model. The branch-to-Null removal rule is an explicit modeling assumption; it is not experimental proof of incrementality.

### 3. Budget simulation

For a chosen total budget:

```text
Current allocation  = Last-Touch share × total budget
Aligned allocation  = Markov share × total budget
Change              = aligned - current
```

The allocation helper preserves total spend to the cent.

### 4. Conversion propensity model

The prediction target is **conversion within 7 days of the observed path prefix**. Seven days was selected as a practical fixed horizon for this one-month dataset: **86.18% of converted modeled journeys convert within 7 days of their last eligible impression**.

Only snapshots at or before **2018-07-24 23:59:30 UTC** are eligible, because later snapshots do not have a complete 7-day observation window before the dataset ends. This removes right-censoring bias from late-month negative labels.

Each eligible customer contributes in-progress path prefixes from the first impression through at most the fifth impression. Features include:

- number of observed touchpoints;
- number of unique channels;
- per-channel counts;
- first and last observed channel;
- ordered touch positions 1–5.

The project uses **three disjoint customer partitions**:

- 60% training;
- 20% validation;
- 20% untouched test.

The same customer can never appear in more than one partition. Logistic Regression and XGBoost are fitted on training data. The winner is selected using **validation PR-AUC**. Both models are then refitted on train + validation data, and final metrics are calculated once on the untouched test set.

The models are trained **without artificial class reweighting**. Probability quality is checked on held-out data with Brier score, log loss, mean probability vs observed rate, calibration gap, and 10-bin expected calibration error (ECE).

---

## Verified attribution results

| Channel | First Touch | Last Touch | Markov |
|---|---:|---:|---:|
| Facebook | 30.57% | 30.87% | **30.45%** |
| Instagram | 13.91% | 13.54% | **17.76%** |
| Online Display | 10.84% | 10.38% | **10.71%** |
| Online Video | 21.37% | 22.79% | **18.84%** |
| Paid Search | 23.31% | 22.42% | **22.24%** |

Baseline Markov conversion probability: **0.043753**, matching the modeled journey conversion rate.

### $100,000 example

| Channel | Last-touch allocation | Markov-aligned allocation | Change |
|---|---:|---:|---:|
| Facebook | $30,867.47 | $30,446.63 | -$420.84 |
| Instagram | $13,537.68 | $17,761.98 | **+$4,224.30** |
| Online Display | $10,384.12 | $10,709.41 | +$325.29 |
| Online Video | $22,792.02 | $18,842.72 | **-$3,949.30** |
| Paid Search | $22,418.71 | $22,239.26 | -$179.45 |

Total spend remains **$100,000.00**.

---

## Verified ML results

Prediction horizon: **7 days**  
Eligible snapshots: **393,810**  
Right-censored late-period snapshots excluded: **78,025**  
Eligible customers: **199,997**

| Model | Validation PR-AUC | Untouched Test ROC-AUC | Untouched Test PR-AUC | Brier | 10-bin ECE |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **0.0690** | 0.6474 | 0.0643 | 0.0350 | 0.0013 |
| XGBoost | 0.0687 | 0.6487 | 0.0655 | 0.0350 | 0.0015 |

**Selected model: Logistic Regression**, because model selection is based only on validation PR-AUC. XGBoost happens to score slightly higher on the untouched test set, but the project correctly does not switch models after seeing test results.

For the selected model on the untouched test set:

- observed 7-day positive rate: **3.6733%**;
- mean predicted probability: **3.7465%**;
- calibration gap: approximately **+0.0732 percentage points**;
- 10-bin ECE: approximately **0.13 percentage points**.

The fixed-horizon design is stronger than the previous observed-until-dataset-end target because every training/evaluation snapshot has the same amount of future outcome time available.

---

## Full-stack components

### FastAPI backend

Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API metadata |
| GET | `/health` | Checks database/model/attribution/metrics readiness; returns HTTP 503 when degraded |
| GET | `/project-summary` | Data and model metrics |
| GET | `/attribution` | Computed attribution table |
| POST | `/budget` | Constant-spend reallocation simulation |
| POST | `/predict` | 7-day conversion propensity for a 1–5-touch path |

Prediction body:

```json
{
  "path": ["Instagram", "Facebook", "Paid Search"]
}
```

### SQLite

The generated database stores:

- customer journeys;
- attribution results;
- baseline budget scenario;
- selected-model feature importance;
- project metrics.

### Streamlit

The dashboard displays generated attribution results, interactive budget reallocation, model benchmarking, and live 7-day path-level conversion propensity. It uses FastAPI when available and local artifacts as a fallback.

### Docker

The container runs as a non-root user. `docker-compose.yml` starts API and dashboard services and waits for API health before starting the dashboard.

### CI

GitHub Actions performs:

1. dependency installation;
2. Python compilation;
3. **full artifact regeneration from the raw CSV**;
4. test execution;
5. Docker image build;
6. Docker Compose smoke test of the API and Streamlit health endpoints.

---

## Project structure

```text
.
├── 1_data_prep.py
├── 2_attribution_models.py
├── 3_budget_simulation.py
├── 4_ml_conversion_model.py
├── 5_dashboard.py
├── 6_build_database.py
├── run_pipeline.py
├── api.py
├── attribution_data.csv
├── artifacts/
├── models/
├── src/marketing_attribution/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── PROJECT_STATUS.md
└── VERIFICATION_REPORT.md
```

---

## Local setup

The Docker/CI reference runtime is **Python 3.11**, and direct dependencies are pinned to the versions used by this verified build.

```bash
cd final_project
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py
```

Start the API:

```bash
uvicorn api:app --reload --port 8000
```

Start the dashboard in a second terminal:

```bash
streamlit run 5_dashboard.py
```

FastAPI documentation is available at `/docs` while the API is running.

## Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

## Tests

```bash
pytest
```

Current verified result: **15 passed**.

---

## Reproducibility

To regenerate all data, attribution, budget, ML, and database artifacts:

```bash
python run_pipeline.py
```

`run_pipeline.py` resolves the project root internally, so it can be called even when the shell's current directory is elsewhere.

---

## Limitations

- Attribution is observational and does not establish causal incrementality.
- The Markov removal operator is a documented counterfactual assumption.
- Conversion rows with no strictly earlier impression are excluded from impression-path attribution.
- Same-timestamp impressions are excluded because their order relative to conversion is ambiguous.
- The raw dataset covers one month. The ML target avoids unequal follow-up by using a fixed 7-day horizon and excluding late snapshots without a complete horizon; it still cannot represent longer-term conversion behavior beyond 7 days.
- The propensity model uses only channel-path information; campaign cost, creative, device, geography, product, customer attributes, and other useful predictors are unavailable.
- The budget module is an attribution-aligned scenario, not an ROI/response-curve optimizer.
- First-order Markov modeling assumes the next state depends only on the current state.

---
