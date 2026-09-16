"""Measure how stable channel attribution is across repeated model runs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def attribution_stability(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize variability in channel credit across repeated runs.

    Expected columns: run_id, channel, attribution_credit.
    Coefficient of variation makes instability comparable across channels.
    """
    required = {"run_id", "channel", "attribution_credit"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if runs["run_id"].nunique() < 2:
        raise ValueError("at least two runs are required")

    summary = (
        runs.groupby("channel", as_index=False)["attribution_credit"]
        .agg(mean_credit="mean", std_credit="std", min_credit="min", max_credit="max")
    )
    denominator = summary["mean_credit"].abs().replace(0.0, np.nan)
    summary["coefficient_of_variation"] = summary["std_credit"] / denominator
    summary["credit_range"] = summary["max_credit"] - summary["min_credit"]
    return summary.sort_values("coefficient_of_variation", ascending=False, na_position="last")


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "run_id": [1, 1, 2, 2, 3, 3],
            "channel": ["search", "social"] * 3,
            "attribution_credit": [0.44, 0.31, 0.47, 0.27, 0.42, 0.35],
        }
    )
    print(attribution_stability(sample).to_string(index=False))
