"""Streamlit frontend. Uses FastAPI when available and local artifacts as a fallback."""
from __future__ import annotations

import json
import os

import joblib
import pandas as pd
import requests
import streamlit as st

from src.marketing_attribution.budget import simulate_budget
from src.marketing_attribution.config import (
    ATTRIBUTION_PATH, BEST_MODEL_PATH, CHANNELS, DATA_METRICS_PATH,
    MAX_PREFIX_TOUCHPOINTS, MODEL_METRICS_PATH, PREDICTION_HORIZON_DAYS,
)
from src.marketing_attribution.ml import predict_path

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Marketing Attribution Intelligence", layout="wide")


def api_get(endpoint: str):
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=1.5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def api_post(endpoint: str, payload: dict):
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


@st.cache_data
def local_summary() -> dict:
    data = json.loads(DATA_METRICS_PATH.read_text(encoding="utf-8")) if DATA_METRICS_PATH.exists() else {}
    model = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8")) if MODEL_METRICS_PATH.exists() else {}
    return {"data": data, "model": model, "channels": CHANNELS}


@st.cache_data
def local_attribution() -> pd.DataFrame:
    if not ATTRIBUTION_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(ATTRIBUTION_PATH)


@st.cache_resource
def local_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact is missing. Run python run_pipeline.py first.")
    return joblib.load(BEST_MODEL_PATH)


summary = api_get("/project-summary") or local_summary()
attr_payload = api_get("/attribution")
attribution = pd.DataFrame(attr_payload) if attr_payload else local_attribution()

if attribution.empty:
    st.error("Generated attribution artifacts are missing. Run `python run_pipeline.py` first.")
    st.stop()

data_metrics = summary.get("data", {})
model_metrics = summary.get("model", {})
horizon_days = int(model_metrics.get("prediction_horizon_days", PREDICTION_HORIZON_DAYS))

st.title("Marketing Attribution & Budget Intelligence")
st.caption(
    "Customer-journey attribution, fixed-spend reallocation simulation, "
    f"and {horizon_days}-day conversion propensity prediction."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw interactions", f"{int(data_metrics.get('raw_rows', 0)):,}")
c2.metric("Modeled journeys", f"{int(data_metrics.get('journeys_modeled', 0)):,}")
c3.metric("Converted journeys", f"{int(data_metrics.get('converted_journeys', 0)):,}")
c4.metric("Channels", str(len(CHANNELS)))

st.subheader("Attribution comparison")
chart_df = attribution.set_index("Channel")[["First_Touch_%", "Last_Touch_%", "Markov_%"]]
st.bar_chart(chart_df)
st.dataframe(
    attribution[["Channel", "First_Touch_%", "Last_Touch_%", "Markov_%", "Removal_Effect"]].round(4),
    use_container_width=True,
    hide_index=True,
)

with st.expander("How the Markov removal model works"):
    st.write(
        "The model learns a first-order transition matrix from converting and non-converting journeys. "
        "For each marketing channel, probability mass that would enter that unavailable channel is redirected "
        "to the Null absorbing state, then the probability of eventually reaching Conversion is recomputed. "
        "The relative drop is the channel's removal effect; positive effects are normalized into attribution shares."
    )
    st.caption(
        "This is observational attribution, not a causal incrementality experiment. The removal rule is an explicit "
        "counterfactual assumption and is documented in the project methodology."
    )

st.subheader("Fixed-spend budget reallocation simulation")
budget_value = st.slider("Total marketing budget ($)", 20_000, 500_000, 100_000, 5_000)
budget_payload = api_post("/budget", {"total_budget": budget_value})
if budget_payload:
    budget_df = pd.DataFrame(budget_payload["allocations"])
else:
    budget_df = simulate_budget(attribution.set_index("Channel"), budget_value).reset_index()

show_budget = budget_df[[
    "Channel", "Current_Budget_Last_Touch", "Attribution_Aligned_Budget", "Budget_Change"
]].copy()
for column in ["Current_Budget_Last_Touch", "Attribution_Aligned_Budget", "Budget_Change"]:
    show_budget[column] = show_budget[column].map(lambda value: f"${value:,.2f}")
st.dataframe(show_budget, use_container_width=True, hide_index=True)
st.info(
    "This preserves total spend and illustrates how allocation changes when Markov attribution replaces last-touch "
    "credit. It is not a causal ROI optimizer because the source data do not contain spend-response curves or experiments."
)

st.subheader(f"In-progress {horizon_days}-day conversion propensity")
best_model = model_metrics.get("best_model", "model")
best_test = model_metrics.get("best_model_test_metrics", {})
auc = best_test.get("roc_auc")
pr_auc = best_test.get("pr_auc")
caption = best_model.replace("_", " ").title()
if auc is not None and pr_auc is not None:
    caption += f" · untouched-test ROC-AUC {auc:.3f} · PR-AUC {pr_auc:.3f}"
st.caption(caption + f" · prefixes capped at {MAX_PREFIX_TOUCHPOINTS} touchpoints")

n_touches = st.slider("Observed touchpoints", 1, MAX_PREFIX_TOUCHPOINTS, 3)
path = []
columns = st.columns(n_touches)
for i, column in enumerate(columns):
    with column:
        default_idx = i % len(CHANNELS)
        path.append(st.selectbox(f"Touch {i + 1}", CHANNELS, index=default_idx, key=f"touch_{i}"))

if st.button("Estimate conversion propensity", type="primary"):
    prediction = api_post("/predict", {"path": path})
    try:
        probability = (
            float(prediction["conversion_probability"])
            if prediction
            else predict_path(local_model(), path, MAX_PREFIX_TOUCHPOINTS)
        )
        st.metric(f"Estimated conversion probability within {horizon_days} days", f"{probability * 100:.1f}%")
        st.progress(probability)
        st.caption(
            f"The estimate is trained only on snapshots with a complete {horizon_days}-day follow-up window. "
            "It reflects patterns in this dataset and is not causal or a guarantee for a new campaign."
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(str(exc))

st.subheader("Model benchmarking")
benchmark_rows = []
validation = model_metrics.get("validation", {})
test = model_metrics.get("test", {})
for name in ["logistic_regression", "xgboost"]:
    if isinstance(test.get(name), dict):
        row = {
            "Model": name.replace("_", " ").title(),
            "Validation PR-AUC": validation.get(name, {}).get("pr_auc"),
            "Test ROC-AUC": test[name].get("roc_auc"),
            "Test PR-AUC": test[name].get("pr_auc"),
            "Brier score": test[name].get("brier_score"),
            "Calibration gap": test[name].get("calibration_gap"),
        }
        benchmark_rows.append(row)
if benchmark_rows:
    st.dataframe(pd.DataFrame(benchmark_rows).round(4), use_container_width=True, hide_index=True)

st.caption(
    "Leakage controls: only strictly pre-conversion impressions enter journeys, features use observed prefixes only, "
    f"late snapshots without a complete {horizon_days}-day outcome window are excluded, and customers are disjoint "
    "across train, validation, and untouched test partitions. Model selection uses validation PR-AUC."
)
