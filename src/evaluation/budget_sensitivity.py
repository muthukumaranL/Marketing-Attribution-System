"""Stress-test channel allocations across budget scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd


def budget_sensitivity(channel_curves: dict[str, list[float]], budgets: list[int]) -> pd.DataFrame:
    """Allocate integer budget units greedily by marginal incremental return.

    Each curve contains cumulative expected return at spend 0, 1, ... N. The
    resulting table shows how allocation changes as the total budget changes.
    """
    if not channel_curves:
        raise ValueError("channel_curves cannot be empty")
    curves = {k: np.asarray(v, dtype=float) for k, v in channel_curves.items()}
    if any(len(v) < 2 for v in curves.values()):
        raise ValueError("Each curve needs at least spend 0 and spend 1")

    rows = []
    for budget in sorted(set(map(int, budgets))):
        if budget < 0:
            raise ValueError("budgets must be non-negative")
        allocation = {name: 0 for name in curves}
        for _ in range(budget):
            gains = {}
            for name, curve in curves.items():
                spend = allocation[name]
                gains[name] = curve[spend + 1] - curve[spend] if spend + 1 < len(curve) else -np.inf
            best = max(gains, key=gains.get)
            if not np.isfinite(gains[best]) or gains[best] <= 0:
                break
            allocation[best] += 1

        total_return = sum(curves[name][spend] for name, spend in allocation.items())
        row = {"budget": budget, "expected_return": float(total_return), **{f"spend_{k}": v for k, v in allocation.items()}}
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    curves = {"search": [0, 9, 17, 23, 27], "social": [0, 7, 13, 18, 22], "email": [0, 5, 8, 10, 11]}
    print(budget_sensitivity(curves, [2, 4, 6, 8]).to_string(index=False))
