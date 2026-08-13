"""Step 4 — train and select leakage-resistant fixed-horizon conversion models."""
from __future__ import annotations

import json
import shutil

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.marketing_attribution.config import (
    BEST_MODEL_PATH, DATA_METRICS_PATH, FEATURE_IMPORTANCE_PATH, JOURNEYS_PATH,
    LR_MODEL_PATH, MAX_PREFIX_TOUCHPOINTS, MODEL_METRICS_PATH,
    PREDICTION_HORIZON_DAYS, RANDOM_STATE, XGB_MODEL_PATH,
)
from src.marketing_attribution.data import load_journeys
from src.marketing_attribution.ml import build_snapshot_dataset, customer_partitions, feature_columns


def evaluate_probabilities(model, X: pd.DataFrame, y: pd.Series) -> dict:
    probs = model.predict_proba(X)[:, 1]
    observed_rate = float(y.mean())
    mean_probability = float(np.mean(probs))

    calibration_frame = pd.DataFrame({"probability": probs, "outcome": y.to_numpy()})
    calibration_frame["bin"] = pd.qcut(calibration_frame["probability"], q=10, duplicates="drop")
    grouped = calibration_frame.groupby("bin", observed=True).agg(
        mean_probability=("probability", "mean"),
        observed_rate=("outcome", "mean"),
        count=("outcome", "size"),
    )
    expected_calibration_error = float(
        ((grouped["mean_probability"] - grouped["observed_rate"]).abs() * grouped["count"]).sum()
        / grouped["count"].sum()
    )

    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "brier_score": float(brier_score_loss(y, probs)),
        "log_loss": float(log_loss(y, probs)),
        "observed_positive_rate": observed_rate,
        "mean_predicted_probability": mean_probability,
        "calibration_gap": float(mean_probability - observed_rate),
        "ece_10_bin": expected_calibration_error,
    }


def make_lr(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)),
    ])


def make_xgb(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", xgb.XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=RANDOM_STATE,
        )),
    ])


def save_feature_importance(model: Pipeline, model_name: str) -> None:
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    estimator = model.named_steps["model"]
    if model_name == "logistic_regression":
        signed = estimator.coef_[0]
        importance = np.abs(signed)
        frame = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
            "signed_effect": signed,
            "model": model_name,
        })
    else:
        frame = pd.DataFrame({
            "feature": feature_names,
            "importance": estimator.feature_importances_,
            "signed_effect": np.nan,
            "model": model_name,
        })
    frame.sort_values("importance", ascending=False).to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def main() -> None:
    journeys = load_journeys(JOURNEYS_PATH)
    data_metrics = json.loads(DATA_METRICS_PATH.read_text(encoding="utf-8"))
    observation_end = pd.Timestamp(data_metrics["data_end"])

    snapshots = build_snapshot_dataset(
        journeys,
        observation_end=observation_end,
        horizon_days=PREDICTION_HORIZON_DAYS,
        max_prefix=MAX_PREFIX_TOUCHPOINTS,
    )
    numeric, categorical = feature_columns(MAX_PREFIX_TOUCHPOINTS)
    feature_cols = numeric + categorical

    # Stratify on the fixed-horizon target among customers that have at least one
    # fully observed snapshot. A customer is positive for splitting if any eligible
    # prefix converts within the configured horizon.
    customer_labels = snapshots.groupby("cookie", as_index=False)["converted"].max()
    train_cookies, validation_cookies, test_cookies = customer_partitions(customer_labels)

    train = snapshots[snapshots["cookie"].isin(train_cookies)].reset_index(drop=True)
    validation = snapshots[snapshots["cookie"].isin(validation_cookies)].reset_index(drop=True)
    test = snapshots[snapshots["cookie"].isin(test_cookies)].reset_index(drop=True)

    X_train, y_train = train[feature_cols], train["converted"].astype(int)
    X_validation, y_validation = validation[feature_cols], validation["converted"].astype(int)
    X_test, y_test = test[feature_cols], test["converted"].astype(int)

    candidates = {
        "logistic_regression": make_lr(numeric, categorical),
        "xgboost": make_xgb(numeric, categorical),
    }

    validation_metrics: dict[str, dict] = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        validation_metrics[name] = evaluate_probabilities(model, X_validation, y_validation)

    # PR-AUC is used for selection because the fixed-horizon outcome is imbalanced.
    best_name = max(validation_metrics, key=lambda name: validation_metrics[name]["pr_auc"])

    train_validation = pd.concat([train, validation], ignore_index=True)
    X_train_validation = train_validation[feature_cols]
    y_train_validation = train_validation["converted"].astype(int)

    final_models = {
        "logistic_regression": make_lr(numeric, categorical),
        "xgboost": make_xgb(numeric, categorical),
    }
    test_metrics: dict[str, dict] = {}
    for name, model in final_models.items():
        model.fit(X_train_validation, y_train_validation)
        test_metrics[name] = evaluate_probabilities(model, X_test, y_test)
        joblib.dump(model, LR_MODEL_PATH if name == "logistic_regression" else XGB_MODEL_PATH)

    best_model = final_models[best_name]
    source = LR_MODEL_PATH if best_name == "logistic_regression" else XGB_MODEL_PATH
    shutil.copy2(source, BEST_MODEL_PATH)
    save_feature_importance(best_model, best_name)

    partitions = [train_cookies, validation_cookies, test_cookies]
    overlap = sum(len(partitions[i].intersection(partitions[j])) for i in range(3) for j in range(i + 1, 3))
    best_test = test_metrics[best_name]

    total_possible_prefixes = int(journeys["num_touchpoints"].clip(upper=MAX_PREFIX_TOUCHPOINTS).sum())
    censored_excluded = int(total_possible_prefixes - len(snapshots))
    eligibility_cutoff = observation_end - pd.Timedelta(days=PREDICTION_HORIZON_DAYS)
    converted_journeys = journeys[journeys["converted"].eq(1)].copy()
    last_touch_delay_days = (
        pd.to_datetime(converted_journeys["conversion_time"], utc=True)
        - pd.to_datetime(converted_journeys["last_touch_time"], utc=True)
    ).dt.total_seconds() / 86400.0
    last_touch_within_horizon_pct = float(
        (last_touch_delay_days <= PREDICTION_HORIZON_DAYS).mean() * 100.0
    )

    metrics = {
        "prediction_target": f"conversion within {PREDICTION_HORIZON_DAYS} days after an in-progress prefix of 1-{MAX_PREFIX_TOUCHPOINTS} ad impressions",
        "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
        "observation_end": observation_end.isoformat(),
        "snapshot_eligibility_cutoff": eligibility_cutoff.isoformat(),
        "max_prefix_touchpoints": MAX_PREFIX_TOUCHPOINTS,
        "snapshot_rows": int(len(snapshots)),
        "right_censored_snapshots_excluded": censored_excluded,
        "converted_journeys_with_last_touch_within_horizon_pct": last_touch_within_horizon_pct,
        "eligible_customers": int(customer_labels["cookie"].nunique()),
        "train_snapshots": int(len(train)),
        "validation_snapshots": int(len(validation)),
        "test_snapshots": int(len(test)),
        "train_customers": int(len(train_cookies)),
        "validation_customers": int(len(validation_cookies)),
        "test_customers": int(len(test_cookies)),
        "customer_overlap_across_partitions": int(overlap),
        "selection_metric": "validation_pr_auc",
        "best_model": best_name,
        "validation": validation_metrics,
        "test": test_metrics,
        "best_model_test_metrics": best_test,
        "probability_calibration_note": "Models are trained without class reweighting; calibration is evaluated on held-out data with Brier score, log loss, calibration gap, and 10-bin ECE.",
        "leakage_control": "strict pre-conversion journey cutoff + prefix-only features + complete fixed-horizon follow-up + disjoint customer train/validation/test partitions",
    }
    MODEL_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"Target: conversion within {PREDICTION_HORIZON_DAYS} days; "
        f"late snapshots without complete follow-up are excluded."
    )
    print(f"Eligible snapshots: {len(snapshots):,}; right-censored excluded: {censored_excluded:,}")
    print(f"Customer overlap across partitions: {overlap}")
    for name in candidates:
        vm = validation_metrics[name]
        tm = test_metrics[name]
        print(
            f"{name}: validation PR-AUC={vm['pr_auc']:.4f}; "
            f"test ROC-AUC={tm['roc_auc']:.4f}; test PR-AUC={tm['pr_auc']:.4f}; "
            f"test Brier={tm['brier_score']:.4f}; calibration gap={tm['calibration_gap']:+.4f}"
        )
    print(f"Selected model (validation PR-AUC): {best_name}")
    print(f"Saved: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
