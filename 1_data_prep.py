"""Step 1 — clean raw events and reconstruct leakage-safe customer journeys."""
import json

from src.marketing_attribution.config import DATA_METRICS_PATH, DATA_PATH, JOURNEYS_PATH
from src.marketing_attribution.data import load_and_clean_raw, reconstruct_journeys, save_journeys


def main() -> None:
    df, raw_metrics = load_and_clean_raw(DATA_PATH)
    journeys, journey_metrics = reconstruct_journeys(df)
    metrics = {**raw_metrics, **journey_metrics}
    save_journeys(journeys, JOURNEYS_PATH)
    DATA_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Raw rows: {metrics['raw_rows']:,}")
    print(f"Duplicates removed: {metrics['duplicate_rows_removed']:,}")
    print(f"Modeled journeys: {metrics['journeys_modeled']:,}")
    print(f"Converted modeled journeys: {metrics['converted_journeys']:,}")
    print(f"Conversions without a strictly prior impression: {metrics['conversions_without_strictly_prior_impression']:,}")
    print(f"Same-timestamp impressions excluded: {metrics['same_timestamp_impressions_excluded']:,}")
    print(f"Post-conversion impressions excluded: {metrics['post_conversion_impressions_excluded']:,}")
    print(f"Saved: {JOURNEYS_PATH}")


if __name__ == "__main__":
    main()
