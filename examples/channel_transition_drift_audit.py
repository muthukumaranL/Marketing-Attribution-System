"""Detect changes in ordered channel-to-channel customer journey transitions."""

from __future__ import annotations

from collections import Counter

import numpy as np


def _distribution(journeys: list[list[str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for journey in journeys:
        counts.update(zip(journey, journey[1:]))
    return counts


def channel_transition_drift(
    reference: list[list[str]],
    candidate: list[list[str]],
    smoothing: float = 1e-6,
) -> dict[str, object]:
    """Compare transition distributions with Jensen-Shannon divergence."""
    if not reference or not candidate or smoothing <= 0:
        raise ValueError("journey sets must be non-empty and smoothing must be positive")
    ref_counts = _distribution(reference)
    cand_counts = _distribution(candidate)
    keys = sorted(set(ref_counts) | set(cand_counts))
    if not keys:
        raise ValueError("journeys must contain at least one channel transition")

    p = np.array([ref_counts[key] + smoothing for key in keys], dtype=float)
    q = np.array([cand_counts[key] + smoothing for key in keys], dtype=float)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log2(p / midpoint)) + 0.5 * np.sum(q * np.log2(q / midpoint))

    shifts = [
        {"transition": f"{a} -> {b}", "reference_share": float(p[i]), "candidate_share": float(q[i]), "share_shift": float(q[i] - p[i])}
        for i, (a, b) in enumerate(keys)
    ]
    shifts.sort(key=lambda row: abs(float(row["share_shift"])), reverse=True)
    return {"js_divergence": float(js), "largest_shifts": shifts[:10]}


if __name__ == "__main__":
    before = [["search", "email", "direct"], ["social", "search", "direct"]]
    after = [["social", "email", "direct"], ["social", "email", "direct"]]
    print(channel_transition_drift(before, after))