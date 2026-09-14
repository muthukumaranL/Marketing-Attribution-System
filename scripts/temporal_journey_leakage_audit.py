"""Detect customer-journey leakage across temporal train/test splits.

Attribution models can leak information when interactions from the same user or
journey appear on both sides of a time split. This audit reports contaminated
entities so evaluation can be repaired before model training.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime


def temporal_journey_leakage_audit(
    rows: Iterable[tuple[str, datetime]],
    *,
    cutoff: datetime,
) -> dict[str, float]:
    user_sides: dict[str, set[str]] = defaultdict(set)
    train_rows = 0
    test_rows = 0

    for user_id, timestamp in rows:
        if not user_id:
            raise ValueError("user_id cannot be empty")
        side = "train" if timestamp < cutoff else "test"
        user_sides[user_id].add(side)
        if side == "train":
            train_rows += 1
        else:
            test_rows += 1

    if not user_sides:
        raise ValueError("rows cannot be empty")

    leaking_users = sum(sides == {"train", "test"} for sides in user_sides.values())
    total_users = len(user_sides)

    return {
        "users": float(total_users),
        "train_rows": float(train_rows),
        "test_rows": float(test_rows),
        "leaking_users": float(leaking_users),
        "user_leakage_rate": leaking_users / total_users,
        "clean_user_share": 1.0 - leaking_users / total_users,
    }


if __name__ == "__main__":
    cutoff = datetime(2026, 7, 1)
    interactions = [
        ("u1", datetime(2026, 6, 20)),
        ("u1", datetime(2026, 7, 3)),
        ("u2", datetime(2026, 6, 25)),
        ("u3", datetime(2026, 7, 4)),
        ("u4", datetime(2026, 6, 29)),
        ("u4", datetime(2026, 7, 2)),
    ]

    print("Temporal journey leakage audit")
    for metric, value in temporal_journey_leakage_audit(interactions, cutoff=cutoff).items():
        print(f"{metric:>22}: {value:.4f}")
