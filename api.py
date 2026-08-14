"""FastAPI backend for attribution, budget simulation, and conversion prediction."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.marketing_attribution.budget import simulate_budget
from src.marketing_attribution.config import (
    ATTRIBUTION_PATH, BEST_MODEL_PATH, CHANNELS, DATA_METRICS_PATH, DB_PATH,
    MAX_PREFIX_TOUCHPOINTS, MODEL_METRICS_PATH, PREDICTION_HORIZON_DAYS,
)
from src.marketing_attribution.db import read_table
from src.marketing_attribution.ml import predict_path

app = FastAPI(
    title="Marketing Attribution Intelligence API",
    version="4.0.0",
    description=(
        "Serves computed attribution, fixed-spend budget simulations, and leakage-resistant "
        "fixed-horizon conversion propensity predictions from the same generated artifacts "
        "used by the dashboard."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class BudgetRequest(BaseModel):
    total_budget: float = Field(gt=0, le=100_000_000)


class PredictionRequest(BaseModel):
    path: list[str] = Field(min_length=1, max_length=MAX_PREFIX_TOUCHPOINTS)

    @field_validator("path")
    @classmethod
    def validate_channels(cls, value: list[str]) -> list[str]:
        invalid = sorted(set(value) - set(CHANNELS))
        if invalid:
            raise ValueError(f"Unknown channel(s): {invalid}")
        return value


@lru_cache(maxsize=1)
def get_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact not found. Run python run_pipeline.py first.")
    return joblib.load(BEST_MODEL_PATH)


def load_json(path: Path, *, required: bool = False) -> dict:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required artifact is missing: {path.name}")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if required and not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path.name}")
    return payload


def load_attribution_frame() -> pd.DataFrame:
    if DB_PATH.exists():
        try:
            frame = read_table("attribution_results")
            if not frame.empty:
                return frame
        except Exception:
            pass
    if not ATTRIBUTION_PATH.exists():
        raise FileNotFoundError("Attribution artifact not found. Run python run_pipeline.py first.")
    frame = pd.read_csv(ATTRIBUTION_PATH)
    if frame.empty:
        raise ValueError("Attribution artifact is empty")
    return frame


@app.get("/")
def root() -> dict:
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> JSONResponse:
    database_ready = False
    model_ready = False
    attribution_ready = False
    metrics_ready = False

    try:
        if DB_PATH.exists():
            frame = read_table("attribution_results")
            database_ready = not frame.empty
    except Exception:
        database_ready = False

    try:
        get_model()
        model_ready = True
    except Exception:
        model_ready = False

    try:
        attribution_ready = not load_attribution_frame().empty
    except Exception:
        attribution_ready = False

    try:
        data_metrics = load_json(DATA_METRICS_PATH, required=True)
        model_metrics = load_json(MODEL_METRICS_PATH, required=True)
        metrics_ready = bool(data_metrics) and bool(model_metrics)
    except Exception:
        metrics_ready = False

    ready = database_ready and model_ready and attribution_ready and metrics_ready
    payload = {
        "status": "ready" if ready else "degraded",
        "ready": ready,
        "database_ready": database_ready,
        "model_ready": model_ready,
        "attribution_ready": attribution_ready,
        "metrics_ready": metrics_ready,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@app.get("/project-summary")
def project_summary() -> dict:
    try:
        return {
            "data": load_json(DATA_METRICS_PATH, required=True),
            "model": load_json(MODEL_METRICS_PATH, required=True),
            "channels": CHANNELS,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Project metrics are unavailable") from exc


@app.get("/attribution")
def attribution() -> list[dict]:
    try:
        return load_attribution_frame().to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Attribution results are unavailable") from exc


@app.post("/budget")
def budget(request: BudgetRequest) -> dict:
    try:
        df = load_attribution_frame().set_index("Channel")
        result = simulate_budget(df, request.total_budget).reset_index()
        return {
            "total_budget": request.total_budget,
            "method": "attribution-aligned fixed-spend simulation",
            "causal_optimization": False,
            "allocations": result.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Budget simulation is unavailable") from exc


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        model = get_model()
        probability = predict_path(model, request.path, MAX_PREFIX_TOUCHPOINTS)
        metrics = load_json(MODEL_METRICS_PATH, required=True)
        horizon_days = int(metrics.get("prediction_horizon_days", PREDICTION_HORIZON_DAYS))
        return {
            "path": request.path,
            "conversion_probability": probability,
            "model": metrics.get("best_model", "unknown"),
            "prediction_horizon_days": horizon_days,
            "prediction_scope": (
                f"conversion within {horizon_days} days after an in-progress prefix of up to "
                f"{MAX_PREFIX_TOUCHPOINTS} impressions"
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Prediction is unavailable") from exc
