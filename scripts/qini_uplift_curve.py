"""Evaluate uplift-model rankings with a simple Qini-style curve.

The utility sorts customers by predicted treatment effect and compares the
cumulative incremental outcome in treated versus control groups. It provides a
model-free way to test whether high-ranked customers are actually more
responsive to treatment.
"""
from __future__ import annotations

import numpy as np


def qini_curve(y, treatment, uplift_score, bins: int = 10) -> dict:
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    score = np.asarray(uplift_score, dtype=float)
    if not (y.shape == treatment.shape == score.shape) or y.ndim != 1 or y.size == 0:
        raise ValueError("y, treatment, and uplift_score must be aligned vectors")
    if np.any((treatment != 0) & (treatment != 1)):
        raise ValueError("treatment must contain only 0 and 1")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    order = np.argsort(-score, kind="stable")
    y = y[order]
    treatment = treatment[order]
    n = len(y)
    rows = []
    for end in np.unique(np.ceil(np.linspace(n / bins, n, bins)).astype(int)):
        ys = y[:end]
        ts = treatment[:end]
        treated = ts == 1
        control = ~treated
        treated_n = int(treated.sum())
        control_n = int(control.sum())
        treated_rate = float(ys[treated].mean()) if treated_n else 0.0
        control_rate = float(ys[control].mean()) if control_n else 0.0
        incremental = (treated_rate - control_rate) * end
        rows.append(
            {
                "fraction_targeted": end / n,
                "count": int(end),
                "treated_rate": treated_rate,
                "control_rate": control_rate,
                "incremental_outcomes": float(incremental),
            }
        )

    x = np.array([0.0] + [r["fraction_targeted"] for r in rows])
    q = np.array([0.0] + [r["incremental_outcomes"] for r in rows])
    random_line = x * q[-1]
    qini_coefficient = float(np.trapz(q - random_line, x))
    return {"qini_coefficient": qini_coefficient, "curve": rows}


if __name__ == "__main__":
    rng = np.random.default_rng(13)
    treatment = rng.binomial(1, 0.5, 1000)
    latent_uplift = rng.normal(0.08, 0.05, 1000)
    base = np.full(1000, 0.08)
    probability = np.clip(base + treatment * latent_uplift, 0, 1)
    outcomes = rng.binomial(1, probability)
    noisy_score = latent_uplift + rng.normal(0, 0.03, 1000)
    print(qini_curve(outcomes, treatment, noisy_score))
