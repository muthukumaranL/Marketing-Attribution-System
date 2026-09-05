"""Probability calibration check for conversion models.

A conversion model can rank users well while still producing poorly calibrated
probabilities. This utility measures Brier score and prints reliability bins.

Run:
    python examples/probability_calibration_check.py
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split


def main() -> None:
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=5,
        weights=[0.94, 0.06],
        random_state=7,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=7
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]

    print(f"Brier score: {brier_score_loss(y_test, prob):.4f}")
    observed, predicted = calibration_curve(y_test, prob, n_bins=8, strategy="quantile")
    print("\nPredicted probability -> observed conversion rate")
    for p_hat, actual in zip(predicted, observed):
        print(f"{p_hat:>7.3f} -> {actual:>7.3f}")


if __name__ == "__main__":
    main()
