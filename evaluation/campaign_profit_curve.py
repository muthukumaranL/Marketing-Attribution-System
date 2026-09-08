"""Choose a conversion-probability threshold using expected campaign profit.

CSV columns:
    y_true,y_prob

Example:
    python evaluation/campaign_profit_curve.py --csv predictions.csv \
        --revenue-per-conversion 120 --contact-cost 4

The script evaluates many probability thresholds and reports the one with the
highest realized profit on labeled evaluation data. This complements ROC-AUC,
PR-AUC, calibration and lift metrics with a directly business-facing decision
criterion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def profit_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    revenue_per_conversion: float,
    contact_cost: float,
) -> pd.DataFrame:
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0/1 labels")
    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_prob must be between 0 and 1")
    if revenue_per_conversion < 0 or contact_cost < 0:
        raise ValueError("economic inputs cannot be negative")

    thresholds = np.linspace(0.01, 0.99, 99)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        targeted = y_prob >= threshold
        contacts = int(targeted.sum())
        conversions = int(y_true[targeted].sum())
        revenue = conversions * revenue_per_conversion
        cost = contacts * contact_cost
        profit = revenue - cost
        rows.append(
            {
                "threshold": float(threshold),
                "contacts": contacts,
                "conversions": conversions,
                "conversion_rate": conversions / contacts if contacts else 0.0,
                "profit": float(profit),
                "profit_per_contact": profit / contacts if contacts else 0.0,
            }
        )
    return pd.DataFrame(rows)


def demo(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = rng.binomial(1, 0.08, size=3_000)
    signal = rng.normal(0, 0.9, size=3_000) + y_true * 1.3
    y_prob = 1 / (1 + np.exp(-(signal - 2.1)))
    return pd.DataFrame({"y_true": y_true, "y_prob": y_prob})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--revenue-per-conversion", type=float, default=100.0)
    parser.add_argument("--contact-cost", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv) if args.csv else demo()
    report = profit_curve(
        frame["y_true"].to_numpy(),
        frame["y_prob"].to_numpy(),
        revenue_per_conversion=args.revenue_per_conversion,
        contact_cost=args.contact_cost,
    )
    best = report.loc[report["profit"].idxmax()]
    print("Best campaign operating point")
    print(best.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nNearby thresholds")
    nearby = report.iloc[max(0, best.name - 2) : best.name + 3]
    print(nearby.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
