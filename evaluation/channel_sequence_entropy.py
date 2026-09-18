"""Measure how diverse customer channel journeys are before attribution modeling."""

from __future__ import annotations

from collections import Counter
from math import log2

import pandas as pd


def journey_sequence_entropy(
    journeys: pd.DataFrame,
    *,
    user_col: str = "user_id",
    channel_col: str = "channel",
    time_col: str = "timestamp",
) -> dict[str, float | int]:
    """Compute entropy and concentration of observed ordered channel sequences."""
    required = {user_col, channel_col, time_col}
    missing = required - set(journeys.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if journeys.empty:
        raise ValueError("journeys must be non-empty")

    ordered = journeys.sort_values([user_col, time_col])
    sequences = ordered.groupby(user_col)[channel_col].apply(
        lambda values: " > ".join(map(str, values))
    )
    counts = Counter(sequences)
    total = sum(counts.values())
    shares = [count / total for count in counts.values()]
    entropy = -sum(p * log2(p) for p in shares if p > 0)
    max_entropy = log2(len(shares)) if len(shares) > 1 else 0.0
    return {
        "journeys": total,
        "unique_sequences": len(counts),
        "sequence_entropy_bits": entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy else 0.0,
        "top_sequence_share": max(shares),
    }


if __name__ == "__main__":
    sample = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 4],
        "channel": ["search", "email", "social", "email", "search", "email", "direct"],
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]),
    })
    print(journey_sequence_entropy(sample))
