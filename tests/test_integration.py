import json

import joblib
from fastapi.testclient import TestClient

from api import app
from src.marketing_attribution.config import BEST_MODEL_PATH, MODEL_METRICS_PATH, PREDICTION_HORIZON_DAYS
from src.marketing_attribution.ml import predict_path


def test_saved_model_loads_and_returns_probability():
    model = joblib.load(BEST_MODEL_PATH)
    probability = predict_path(model, ["Instagram", "Facebook"])
    assert 0.0 <= probability <= 1.0


def test_api_smoke_endpoints():
    client = TestClient(app)
    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ready"] is True
    assert health.json()["metrics_ready"] is True
    assert client.get("/project-summary").status_code == 200
    attribution = client.get("/attribution")
    assert attribution.status_code == 200
    assert len(attribution.json()) == 5
    budget = client.post("/budget", json={"total_budget": 100000})
    assert budget.status_code == 200
    assert abs(sum(row["Attribution_Aligned_Budget"] for row in budget.json()["allocations"]) - 100000) < 0.01
    prediction = client.post("/predict", json={"path": ["Instagram", "Facebook"]})
    assert prediction.status_code == 200
    assert 0 <= prediction.json()["conversion_probability"] <= 1
    assert prediction.json()["prediction_horizon_days"] == PREDICTION_HORIZON_DAYS
    assert f"within {PREDICTION_HORIZON_DAYS} days" in prediction.json()["prediction_scope"]


def test_api_validation_errors():
    client = TestClient(app)
    assert client.post("/predict", json={"path": []}).status_code == 422
    assert client.post("/predict", json={"path": ["Unknown"]}).status_code == 422
    assert client.post("/predict", json={"path": ["Facebook"] * 6}).status_code == 422
    assert client.post("/budget", json={"total_budget": 0}).status_code == 422


def test_model_metrics_report_fixed_horizon_untouched_test_and_calibration():
    metrics = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["selection_metric"] == "validation_pr_auc"
    assert metrics["prediction_horizon_days"] == PREDICTION_HORIZON_DAYS
    assert metrics["right_censored_snapshots_excluded"] > 0
    assert metrics["customer_overlap_across_partitions"] == 0
    best = metrics["best_model_test_metrics"]
    assert "brier_score" in best
    assert abs(best["calibration_gap"]) < 0.02
    assert best["ece_10_bin"] < 0.02


def test_health_returns_503_when_required_metrics_are_missing(monkeypatch, tmp_path):
    import api as api_module

    missing = tmp_path / "missing_model_metrics.json"
    monkeypatch.setattr(api_module, "MODEL_METRICS_PATH", missing)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["metrics_ready"] is False
