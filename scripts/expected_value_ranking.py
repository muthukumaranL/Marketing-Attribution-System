"""Rank prospects by expected incremental campaign value.

Expected value combines conversion probability, incremental treatment effect,
conversion margin, and contact cost. This makes targeting decisions closer to
business impact than probability ranking alone.
"""
from __future__ import annotations

import numpy as np


def rank_by_expected_value(ids, conversion_probability, uplift, margin, contact_cost):
    ids = np.asarray(ids)
    p = np.asarray(conversion_probability, dtype=float)
    u = np.asarray(uplift, dtype=float)
    m = np.asarray(margin, dtype=float)
    c = np.asarray(contact_cost, dtype=float)
    if not (ids.shape == p.shape == u.shape == m.shape == c.shape):
        raise ValueError("all inputs must have the same shape")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("conversion probabilities must be in [0, 1]")

    incremental_prob = np.clip(p * u, 0.0, 1.0)
    expected_value = incremental_prob * m - c
    order = np.argsort(-expected_value)
    return [
        {
            "id": str(ids[i]),
            "expected_value": float(expected_value[i]),
            "incremental_probability": float(incremental_prob[i]),
            "target": bool(expected_value[i] > 0),
        }
        for i in order
    ]


if __name__ == "__main__":
    ranked = rank_by_expected_value(
        ["A", "B", "C", "D"],
        [0.12, 0.30, 0.08, 0.22],
        [0.25, 0.05, 0.40, 0.18],
        [120, 120, 120, 120],
        [1.5, 1.5, 1.5, 1.5],
    )
    for row in ranked:
        print(row)
