from __future__ import annotations

import pandas as pd


def _allocate_exact(weights: pd.Series, total_budget: float) -> pd.Series:
    weights = weights.clip(lower=0).astype(float)
    if weights.sum() <= 0:
        raise ValueError("Budget weights must contain a positive value")
    raw = weights / weights.sum() * float(total_budget)
    rounded = raw.round(2)
    difference = round(float(total_budget) - float(rounded.sum()), 2)
    if abs(difference) >= 0.01:
        rounded.loc[raw.idxmax()] += difference
    return rounded


def simulate_budget(comparison: pd.DataFrame, total_budget: float) -> pd.DataFrame:
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    current = _allocate_exact(comparison["Last_Touch_%"], total_budget)
    aligned = _allocate_exact(comparison["Markov_%"], total_budget)
    result = comparison[["Last_Touch_%", "Markov_%"]].copy()
    result["Current_Budget_Last_Touch"] = current
    result["Attribution_Aligned_Budget"] = aligned
    result["Budget_Change"] = aligned - current
    result["Budget_Change_%"] = (result["Budget_Change"] / current.replace(0, pd.NA) * 100).fillna(0.0)
    result.index.name = "Channel"
    return result
