import pandas as pd
import pytest

from src.marketing_attribution.ml import (
    build_snapshot_dataset, customer_partitions, path_to_feature_row,
)


def test_snapshots_use_prefixes_fixed_horizon_and_exclude_censored_rows():
    journeys = pd.DataFrame({
        "cookie": ["x", "y"],
        "path": [
            ["Facebook", "Instagram", "Paid Search"],
            ["Facebook"],
        ],
        "touch_times": [
            [
                pd.Timestamp("2026-01-01T10:00:00Z"),
                pd.Timestamp("2026-01-04T10:00:00Z"),
                pd.Timestamp("2026-01-08T10:00:00Z"),
            ],
            [pd.Timestamp("2026-01-09T10:00:00Z")],
        ],
        "conversion_time": [
            pd.Timestamp("2026-01-10T10:00:00Z"),
            pd.NaT,
        ],
    })
    snapshots = build_snapshot_dataset(
        journeys,
        observation_end=pd.Timestamp("2026-01-10T23:59:59Z"),
        horizon_days=7,
        max_prefix=3,
    )

    # Only the Jan-1 snapshot has a full 7-day follow-up window. Its conversion
    # happens 9 days later, so its fixed-horizon label is 0.
    assert len(snapshots) == 1
    assert snapshots.iloc[0]["last_channel"] == "Facebook"
    assert snapshots.iloc[0]["converted"] == 0
    assert snapshots.iloc[0]["snapshot_time"] == pd.Timestamp("2026-01-01T10:00:00Z")


def test_snapshot_positive_when_conversion_occurs_within_horizon():
    journeys = pd.DataFrame({
        "cookie": ["x"],
        "path": [["Facebook", "Instagram"]],
        "touch_times": [[
            pd.Timestamp("2026-01-01T10:00:00Z"),
            pd.Timestamp("2026-01-02T10:00:00Z"),
        ]],
        "conversion_time": [pd.Timestamp("2026-01-04T10:00:00Z")],
    })
    snapshots = build_snapshot_dataset(
        journeys,
        observation_end=pd.Timestamp("2026-01-20T00:00:00Z"),
        horizon_days=7,
        max_prefix=3,
    )
    assert snapshots["converted"].tolist() == [1, 1]
    assert snapshots.iloc[1]["last_channel"] == "Instagram"


def test_customer_partitions_are_disjoint_and_complete():
    labels = pd.DataFrame({
        "cookie": [f"c{i}" for i in range(100)],
        "converted": [0] * 80 + [1] * 20,
    })
    train, validation, test = customer_partitions(labels)
    assert not train.intersection(validation)
    assert not train.intersection(test)
    assert not validation.intersection(test)
    assert train | validation | test == set(labels["cookie"])
    assert len(train) == 60
    assert len(validation) == 20
    assert len(test) == 20


def test_feature_row_rejects_empty_path():
    with pytest.raises(ValueError, match="At least one"):
        path_to_feature_row([])
