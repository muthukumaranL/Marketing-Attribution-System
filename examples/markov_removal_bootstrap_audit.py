"""Quantify uncertainty in channel removal effects across bootstrap attribution runs."""

from __future__ import annotations

import numpy as np


def removal_effect_confidence(
    channel_names: list[str],
    removal_effect_runs: np.ndarray,
) -> list[dict[str, float | str]]:
    """Return bootstrap intervals and sign stability for channel removal effects."""
    effects = np.asarray(removal_effect_runs, dtype=float)
    if effects.ndim != 2 or effects.shape[1] != len(channel_names) or effects.shape[0] < 2:
        raise ValueError("removal_effect_runs must be runs x channels with at least two runs")
    if len(set(channel_names)) != len(channel_names) or not np.all(np.isfinite(effects)):
        raise ValueError("channel_names must be unique and effects must be finite")

    rows: list[dict[str, float | str]] = []
    for idx, channel in enumerate(channel_names):
        values = effects[:, idx]
        mean = float(np.mean(values))
        rows.append({
            "channel": channel,
            "mean_removal_effect": mean,
            "ci_025": float(np.quantile(values, 0.025)),
            "ci_975": float(np.quantile(values, 0.975)),
            "positive_effect_probability": float(np.mean(values > 0)),
            "sign_stability": float(max(np.mean(values >= 0), np.mean(values <= 0))),
        })
    return sorted(rows, key=lambda row: abs(float(row["mean_removal_effect"])), reverse=True)


if __name__ == "__main__":
    runs = np.array([[0.22, 0.10, 0.04], [0.25, 0.08, -0.01], [0.19, 0.12, 0.02], [0.24, 0.09, 0.00]])
    print(removal_effect_confidence(["search", "email", "social"], runs))
