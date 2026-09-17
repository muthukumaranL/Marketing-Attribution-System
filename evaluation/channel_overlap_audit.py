"""Audit audience overlap between marketing channels."""

from __future__ import annotations

from itertools import combinations

import pandas as pd


def channel_audience_overlap(touches: pd.DataFrame) -> pd.DataFrame:
    """Measure pairwise Jaccard overlap of users reached by each channel."""
    required = {"user_id", "channel"}
    missing = required - set(touches.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    audiences = {
        str(channel): set(group["user_id"].dropna().tolist())
        for channel, group in touches.groupby("channel")
    }
    rows = []
    for left, right in combinations(sorted(audiences), 2):
        a, b = audiences[left], audiences[right]
        union = a | b
        intersection = a & b
        rows.append({
            "channel_a": left,
            "channel_b": right,
            "shared_users": len(intersection),
            "union_users": len(union),
            "jaccard_overlap": len(intersection) / len(union) if union else 0.0,
        })
    return pd.DataFrame(rows).sort_values(
        "jaccard_overlap", ascending=False, ignore_index=True
    ) if rows else pd.DataFrame(columns=[
        "channel_a", "channel_b", "shared_users", "union_users", "jaccard_overlap"
    ])


if __name__ == "__main__":
    sample = pd.DataFrame({
        "user_id": [1, 2, 3, 2, 3, 4, 4, 5],
        "channel": ["search"] * 3 + ["social"] * 3 + ["email"] * 2,
    })
    print(channel_audience_overlap(sample).to_string(index=False))
