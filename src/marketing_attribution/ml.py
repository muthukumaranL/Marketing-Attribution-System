from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CHANNELS, MAX_PREFIX_TOUCHPOINTS, PREDICTION_HORIZON_DAYS, RANDOM_STATE


def path_to_feature_row(path: Sequence[str], max_prefix: int = MAX_PREFIX_TOUCHPOINTS) -> dict:
    path = list(path)[:max_prefix]
    if not path:
        raise ValueError("At least one touchpoint is required")
    invalid = sorted(set(path) - set(CHANNELS))
    if invalid:
        raise ValueError(f"Unknown channels: {invalid}")

    row: dict[str, object] = {
        "num_touchpoints": len(path),
        "num_unique_channels": len(set(path)),
        "first_channel": path[0],
        "last_channel": path[-1],
    }
    for channel in CHANNELS:
        row[f"count_{channel}"] = path.count(channel)
    for i in range(max_prefix):
        row[f"touch_{i + 1}"] = path[i] if i < len(path) else "<NONE>"
    return row


def build_snapshot_dataset(
    journeys: pd.DataFrame,
    observation_end: str | pd.Timestamp,
    horizon_days: int = PREDICTION_HORIZON_DAYS,
    max_prefix: int = MAX_PREFIX_TOUCHPOINTS,
) -> pd.DataFrame:
    """Create leakage-safe fixed-horizon in-progress journey snapshots.

    A snapshot is eligible only when its full prediction horizon is observable in the
    source data. This avoids right-censoring late-period negatives. The binary target
    is 1 only when the customer's first conversion occurs strictly after the snapshot
    and no later than ``horizon_days`` after it.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    required = {"cookie", "path", "touch_times", "conversion_time"}
    missing = required.difference(journeys.columns)
    if missing:
        raise ValueError(f"Missing journey columns required for snapshots: {sorted(missing)}")

    obs_end = pd.Timestamp(observation_end)
    if obs_end.tzinfo is None:
        obs_end = obs_end.tz_localize("UTC")
    horizon = pd.Timedelta(days=horizon_days)
    eligibility_cutoff = obs_end - horizon

    records: list[dict] = []
    for row in journeys.itertuples(index=False):
        path = list(row.path)
        touch_times = [pd.Timestamp(value) for value in row.touch_times]
        conversion_time = pd.Timestamp(row.conversion_time) if pd.notna(row.conversion_time) else pd.NaT
        upper = min(len(path), max_prefix)

        for prefix_len in range(1, upper + 1):
            snapshot_time = touch_times[prefix_len - 1]
            if snapshot_time > eligibility_cutoff:
                continue

            horizon_end = snapshot_time + horizon
            converted_within_horizon = int(
                pd.notna(conversion_time)
                and conversion_time > snapshot_time
                and conversion_time <= horizon_end
            )

            record = path_to_feature_row(path[:prefix_len], max_prefix=max_prefix)
            record["cookie"] = row.cookie
            record["converted"] = converted_within_horizon
            record["snapshot_time"] = snapshot_time
            record["horizon_end"] = horizon_end
            records.append(record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No eligible snapshots remain after applying the prediction horizon")
    return frame


def feature_columns(max_prefix: int = MAX_PREFIX_TOUCHPOINTS) -> tuple[list[str], list[str]]:
    numeric = ["num_touchpoints", "num_unique_channels", *[f"count_{c}" for c in CHANNELS]]
    categorical = ["first_channel", "last_channel", *[f"touch_{i + 1}" for i in range(max_prefix)]]
    return numeric, categorical


def customer_partitions(
    customer_labels: pd.DataFrame,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[set[str], set[str], set[str]]:
    """Return stratified customer-level train/validation/test partitions.

    ``customer_labels`` must contain ``cookie`` and binary ``converted`` columns with
    one target label per customer. Splitting at customer level prevents snapshot rows
    from the same person appearing in multiple partitions.
    """
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1")
    if not {"cookie", "converted"}.issubset(customer_labels.columns):
        raise ValueError("customer_labels must contain cookie and converted columns")

    customers = customer_labels[["cookie", "converted"]].drop_duplicates("cookie")
    if customers["converted"].nunique() < 2:
        raise ValueError("Both positive and negative customers are required for stratified splitting")

    train_val, test = train_test_split(
        customers,
        test_size=test_size,
        random_state=random_state,
        stratify=customers["converted"],
    )
    validation_fraction_of_train_val = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=validation_fraction_of_train_val,
        random_state=random_state + 1,
        stratify=train_val["converted"],
    )
    return set(train["cookie"]), set(validation["cookie"]), set(test["cookie"])


def predict_path(model, path: Sequence[str], max_prefix: int = MAX_PREFIX_TOUCHPOINTS) -> float:
    row = pd.DataFrame([path_to_feature_row(path, max_prefix=max_prefix)])
    prob = float(model.predict_proba(row)[:, 1][0])
    return float(np.clip(prob, 0.0, 1.0))
