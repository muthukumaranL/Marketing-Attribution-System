"""Run the complete reproducible data/attribution/ML/database pipeline."""
from __future__ import annotations

import subprocess
import sys

from src.marketing_attribution.config import ROOT

STEPS = [
    [sys.executable, "1_data_prep.py"],
    [sys.executable, "2_attribution_models.py"],
    [sys.executable, "3_budget_simulation.py"],
    [sys.executable, "4_ml_conversion_model.py"],
    [sys.executable, "6_build_database.py"],
]


def main() -> None:
    for command in STEPS:
        print("\n>", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
