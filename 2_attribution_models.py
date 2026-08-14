"""Step 2 — First/Last Touch and documented Markov removal-effect attribution."""
from src.marketing_attribution.attribution import build_attribution_comparison
from src.marketing_attribution.config import (
    ATTRIBUTION_PATH, JOURNEYS_PATH, MARKOV_DETAILS_PATH, TRANSITION_MATRIX_PATH,
)
from src.marketing_attribution.data import load_journeys


def main() -> None:
    journeys = load_journeys(JOURNEYS_PATH)
    comparison, details, transition_probs, baseline = build_attribution_comparison(journeys)
    comparison.to_csv(ATTRIBUTION_PATH)
    details.to_csv(MARKOV_DETAILS_PATH)
    transition_probs.to_csv(TRANSITION_MATRIX_PATH)

    print(f"Baseline Markov conversion probability: {baseline:.6f}")
    print("\nAttribution comparison (%):")
    print(comparison[["First_Touch_%", "Last_Touch_%", "Markov_%"]].round(2))
    print(f"\nSaved: {ATTRIBUTION_PATH}")


if __name__ == "__main__":
    main()
