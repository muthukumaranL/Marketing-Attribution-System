"""Lift and cumulative-gain diagnostic for conversion models.

ROC-AUC describes ranking quality globally, but marketers often care about how
many conversions are captured in the highest-scored audience segments. This
example reports cumulative gain and lift by score decile.

Run:
    pip install numpy pandas scikit-learn
    python examples/lift_gain_diagnostic.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def lift_table(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Return cumulative gain and lift from highest to lowest predicted score."""
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    total_positives = int(y_true.sum())
    if total_positives == 0:
        raise ValueError("at least one positive outcome is required")

    frame = pd.DataFrame({"actual": y_true, "score": y_score}).sort_values("score", ascending=False)
    frame["bucket"] = pd.qcut(
        np.arange(len(frame)),
        q=bins,
        labels=[f"D{i}" for i in range(1, bins + 1)],
    )

    grouped = (
        frame.groupby("bucket", observed=True)
        .agg(records=("actual", "size"), conversions=("actual", "sum"), mean_score=("score", "mean"))
        .reset_index()
    )
    grouped["cum_records"] = grouped["records"].cumsum()
    grouped["cum_conversions"] = grouped["conversions"].cumsum()
    grouped["population_share"] = grouped["cum_records"] / len(frame)
    grouped["gain"] = grouped["cum_conversions"] / total_positives
    grouped["lift"] = grouped["gain"] / grouped["population_share"]
    return grouped


def main() -> None:
    X, y = make_classification(
        n_samples=7000,
        n_features=14,
        n_informative=7,
        weights=[0.95, 0.05],
        class_sep=1.0,
        random_state=21,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=21
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    table = lift_table(y_test, scores)
    display = table[["bucket", "records", "conversions", "mean_score", "gain", "lift"]].copy()
    print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    top_decile = table.iloc[0]
    print(
        f"\nTop 10% captures {top_decile['gain']:.1%} of conversions "
        f"at {top_decile['lift']:.2f}x random-selection lift."
    )


if __name__ == "__main__":
    main()
