from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import CHANNELS

REQUIRED_COLUMNS = {
    "cookie", "time", "interaction", "conversion", "conversion_value", "channel"
}


def _validate_required_values(df: pd.DataFrame) -> None:
    missing_values = {column: int(df[column].isna().sum()) for column in REQUIRED_COLUMNS if df[column].isna().any()}
    if missing_values:
        raise ValueError(f"Null values found in required columns: {missing_values}")

    cookie_text = df["cookie"].astype(str).str.strip()
    if cookie_text.eq("").any():
        raise ValueError("Blank cookie identifiers are not allowed")


def load_and_clean_raw(path: str | Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    raw_rows = len(df)
    _validate_required_values(df)

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    invalid_times = int(df["time"].isna().sum())
    if invalid_times:
        raise ValueError(f"Found {invalid_times} invalid timestamps")

    df["conversion"] = pd.to_numeric(df["conversion"], errors="raise")
    if not df["conversion"].isin([0, 1]).all():
        bad = sorted(df.loc[~df["conversion"].isin([0, 1]), "conversion"].unique().tolist())
        raise ValueError(f"Conversion flag must be 0/1; found: {bad}")
    df["conversion"] = df["conversion"].astype(int)

    df["conversion_value"] = pd.to_numeric(df["conversion_value"], errors="raise").astype(float)
    if (df["conversion_value"] < 0).any():
        raise ValueError("conversion_value must be non-negative")

    invalid_channels = sorted(set(df["channel"]) - set(CHANNELS))
    if invalid_channels:
        raise ValueError(f"Unexpected channels: {invalid_channels}")

    allowed_interactions = {"impression", "conversion"}
    invalid_interactions = sorted(set(df["interaction"]) - allowed_interactions)
    if invalid_interactions:
        raise ValueError(f"Unexpected interaction types: {invalid_interactions}")

    inconsistent = (
        (df["interaction"].eq("impression") & df["conversion"].ne(0))
        | (df["interaction"].eq("conversion") & df["conversion"].ne(1))
    )
    if inconsistent.any():
        raise ValueError(f"Found {int(inconsistent.sum())} rows with inconsistent interaction/conversion flags")

    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    df = df.sort_values(["cookie", "time"], kind="stable").reset_index(drop=True)

    metrics = {
        "raw_rows": raw_rows,
        "clean_rows": int(len(df)),
        "duplicate_rows_removed": duplicate_rows,
        "unique_customers_raw": int(df["cookie"].nunique()),
        "conversion_events": int(df["conversion"].sum()),
        "unique_converting_customers": int(df.loc[df["conversion"].eq(1), "cookie"].nunique()),
        "channels": CHANNELS,
        "data_start": df["time"].min().isoformat(),
        "data_end": df["time"].max().isoformat(),
    }
    return df, metrics


def reconstruct_journeys(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build leakage-safe impression journeys using a strict pre-conversion cutoff.

    Events are sorted chronologically. For converting customers, only impressions
    strictly earlier than the first conversion timestamp are eligible. Same-second
    impressions are excluded because the source does not provide a deterministic
    within-timestamp event order; including them could expose outcome-time information.
    """
    conversion_rows = df[df["conversion"].eq(1)].copy()
    first_conversion = conversion_rows.groupby("cookie", sort=False)["time"].min().rename("conversion_time")
    conversion_value = conversion_rows.groupby("cookie", sort=False)["conversion_value"].first().rename("conversion_value")

    impressions = df[df["interaction"].eq("impression")].copy()
    impressions = impressions.join(first_conversion, on="cookie")

    same_time_mask = impressions["conversion_time"].notna() & impressions["time"].eq(impressions["conversion_time"])
    post_conversion_mask = impressions["conversion_time"].notna() & impressions["time"].gt(impressions["conversion_time"])
    eligible = impressions["conversion_time"].isna() | impressions["time"].lt(impressions["conversion_time"])
    impressions = impressions.loc[eligible].copy()

    grouped = impressions.groupby("cookie", sort=False).agg(
        path=("channel", list),
        touch_times=("time", list),
        first_touch_time=("time", "min"),
        last_touch_time=("time", "max"),
    )
    grouped = grouped.join(first_conversion).join(conversion_value)
    grouped["converted"] = grouped["conversion_time"].notna().astype(int)
    grouped["conversion_value"] = grouped["conversion_value"].fillna(0.0)
    grouped["num_touchpoints"] = grouped["path"].map(len)
    grouped["num_unique_channels"] = grouped["path"].map(lambda p: len(set(p)))
    grouped = grouped.reset_index()

    converted_customers = int(first_conversion.size)
    converted_modeled = int(grouped["converted"].sum())

    metrics = {
        "journeys_modeled": int(len(grouped)),
        "converted_journeys": converted_modeled,
        "converting_customers_total": converted_customers,
        "conversions_without_strictly_prior_impression": int(converted_customers - converted_modeled),
        "journey_conversion_rate": float(grouped["converted"].mean()),
        "avg_touchpoints_converted": float(grouped.loc[grouped["converted"].eq(1), "num_touchpoints"].mean()),
        "avg_touchpoints_nonconverted": float(grouped.loc[grouped["converted"].eq(0), "num_touchpoints"].mean()),
        "same_timestamp_impressions_excluded": int(same_time_mask.sum()),
        "post_conversion_impressions_excluded": int(post_conversion_mask.sum()),
    }
    return grouped, metrics


def save_journeys(journeys: pd.DataFrame, path: str | Path) -> None:
    out = journeys.copy()
    out["path_json"] = out["path"].map(json.dumps)
    out["touch_times_json"] = out["touch_times"].map(
        lambda values: json.dumps([pd.Timestamp(v).isoformat() for v in values])
    )
    out = out.drop(columns=["path", "touch_times"])
    out.to_csv(path, index=False)


def load_journeys(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["first_touch_time", "last_touch_time", "conversion_time"])
    df["path"] = df["path_json"].map(json.loads)
    df["touch_times"] = df["touch_times_json"].map(
        lambda value: [pd.Timestamp(item) for item in json.loads(value)]
    )
    return df


def collapse_consecutive(path: Iterable[str]) -> list[str]:
    collapsed: list[str] = []
    for channel in path:
        if not collapsed or collapsed[-1] != channel:
            collapsed.append(channel)
    return collapsed
