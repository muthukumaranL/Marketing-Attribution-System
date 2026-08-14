import pandas as pd
import pytest

from src.marketing_attribution.data import load_and_clean_raw, reconstruct_journeys


def test_reconstruct_journey_uses_strict_pre_conversion_cutoff():
    df = pd.DataFrame([
        {"cookie": "a", "time": pd.Timestamp("2026-01-01T10:00:00Z"), "interaction": "impression", "conversion": 0, "conversion_value": 0, "channel": "Facebook"},
        {"cookie": "a", "time": pd.Timestamp("2026-01-01T11:00:00Z"), "interaction": "impression", "conversion": 0, "conversion_value": 0, "channel": "Instagram"},
        {"cookie": "a", "time": pd.Timestamp("2026-01-01T11:00:00Z"), "interaction": "conversion", "conversion": 1, "conversion_value": 10, "channel": "Instagram"},
        {"cookie": "a", "time": pd.Timestamp("2026-01-01T12:00:00Z"), "interaction": "impression", "conversion": 0, "conversion_value": 0, "channel": "Paid Search"},
        {"cookie": "b", "time": pd.Timestamp("2026-01-01T10:00:00Z"), "interaction": "impression", "conversion": 0, "conversion_value": 0, "channel": "Paid Search"},
    ]).sort_values(["cookie", "time"], kind="stable")
    journeys, metrics = reconstruct_journeys(df)
    a = journeys.set_index("cookie").loc["a"]
    assert a["path"] == ["Facebook"]
    assert int(a["converted"]) == 1
    assert metrics["same_timestamp_impressions_excluded"] == 1
    assert metrics["post_conversion_impressions_excluded"] == 1


def test_cleaner_rejects_null_required_values(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([
        {"cookie": "a", "time": "2026-01-01T10:00:00Z", "interaction": "impression", "conversion": 0, "conversion_value": 0, "channel": None},
    ]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Null values"):
        load_and_clean_raw(path)


def test_cleaner_rejects_inconsistent_conversion_flag(tmp_path):
    path = tmp_path / "bad_flag.csv"
    pd.DataFrame([
        {"cookie": "a", "time": "2026-01-01T10:00:00Z", "interaction": "impression", "conversion": 1, "conversion_value": 0, "channel": "Facebook"},
    ]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="inconsistent"):
        load_and_clean_raw(path)
