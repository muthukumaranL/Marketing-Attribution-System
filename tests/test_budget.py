import pandas as pd
import pytest

from src.marketing_attribution.budget import simulate_budget


def test_budget_simulation_preserves_total_spend_to_cent():
    comparison = pd.DataFrame(
        {"Last_Touch_%": [60, 40], "Markov_%": [45, 55]},
        index=["A", "B"],
    )
    result = simulate_budget(comparison, 100_000.01)
    assert result["Current_Budget_Last_Touch"].sum() == pytest.approx(100_000.01, abs=0.001)
    assert result["Attribution_Aligned_Budget"].sum() == pytest.approx(100_000.01, abs=0.001)
    assert result["Budget_Change"].sum() == pytest.approx(0.0, abs=0.001)
