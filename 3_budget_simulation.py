"""Step 3 — fixed-spend attribution-aligned budget reallocation simulation."""
import argparse

import pandas as pd

from src.marketing_attribution.budget import simulate_budget
from src.marketing_attribution.config import ATTRIBUTION_PATH, BUDGET_PATH


def main(total_budget: float = 100_000) -> None:
    comparison = pd.read_csv(ATTRIBUTION_PATH, index_col="Channel")
    result = simulate_budget(comparison, total_budget)
    result.to_csv(BUDGET_PATH)
    print(f"Attribution-aligned simulation at ${total_budget:,.2f} total spend")
    print(result[["Current_Budget_Last_Touch", "Attribution_Aligned_Budget", "Budget_Change"]].round(2))
    print("\nNote: this is an allocation simulation, not a causal ROI/response-curve optimizer.")
    print(f"Saved: {BUDGET_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=100_000)
    args = parser.parse_args()
    main(args.budget)
