"""Allocate a fixed marketing budget across channels with diminishing returns.

Each channel supplies a sequence of incremental expected conversions for equal
spend units. A greedy marginal-gain allocator repeatedly funds the best next
unit, making saturation explicit instead of assuming constant channel ROI.
"""
from __future__ import annotations


def allocate_budget(incremental_returns: dict[str, list[float]], budget_units: int) -> dict:
    if budget_units < 0:
        raise ValueError("budget_units cannot be negative")
    if not incremental_returns:
        raise ValueError("at least one channel is required")

    curves: dict[str, list[float]] = {}
    for channel, values in incremental_returns.items():
        curve = [float(v) for v in values]
        if any(v < 0 for v in curve):
            raise ValueError("incremental returns must be non-negative")
        # Marginal response curves should not improve as spend rises.
        if any(curve[i] < curve[i + 1] for i in range(len(curve) - 1)):
            raise ValueError(f"returns for {channel} must be non-increasing")
        curves[channel] = curve

    allocation = {channel: 0 for channel in curves}
    expected_return = 0.0
    decisions = []

    for unit in range(budget_units):
        candidates = []
        for channel, curve in curves.items():
            index = allocation[channel]
            if index < len(curve):
                candidates.append((curve[index], channel))
        if not candidates:
            break
        marginal_gain, channel = max(candidates, key=lambda item: (item[0], item[1]))
        allocation[channel] += 1
        expected_return += marginal_gain
        decisions.append({
            "budget_unit": unit + 1,
            "channel": channel,
            "marginal_expected_return": marginal_gain,
        })

    return {
        "allocated_units": sum(allocation.values()),
        "allocation": allocation,
        "expected_incremental_conversions": expected_return,
        "decisions": decisions,
    }


if __name__ == "__main__":
    response_curves = {
        "search": [9.2, 7.8, 6.0, 4.1],
        "social": [7.5, 6.9, 5.4, 3.8],
        "email": [5.6, 4.8, 3.2],
    }
    print(allocate_budget(response_curves, budget_units=7))
