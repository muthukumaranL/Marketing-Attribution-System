from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from .config import (
    ATTRIBUTION_PATH, BUDGET_PATH, DATA_METRICS_PATH, DB_PATH,
    FEATURE_IMPORTANCE_PATH, MODEL_METRICS_PATH, JOURNEYS_PATH,
)


def build_database(db_path: str | Path = DB_PATH) -> Path:
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        journeys = pd.read_csv(JOURNEYS_PATH)
        journeys[[
            "cookie", "path_json", "converted", "conversion_value", "num_touchpoints",
            "num_unique_channels", "first_touch_time", "last_touch_time", "conversion_time"
        ]].to_sql("journeys", conn, index=False, if_exists="replace", chunksize=5000)
        conn.execute("CREATE INDEX idx_journeys_converted ON journeys(converted)")

        attribution = pd.read_csv(ATTRIBUTION_PATH)
        attribution.to_sql("attribution_results", conn, index=False, if_exists="replace")

        budget = pd.read_csv(BUDGET_PATH)
        budget.to_sql("budget_baseline", conn, index=False, if_exists="replace")

        if FEATURE_IMPORTANCE_PATH.exists():
            pd.read_csv(FEATURE_IMPORTANCE_PATH).to_sql(
                "feature_importance", conn, index=False, if_exists="replace"
            )

        rows = []
        for section, path in [("data", DATA_METRICS_PATH), ("model", MODEL_METRICS_PATH)]:
            if path.exists():
                payload = json.loads(path.read_text())
                for key, value in payload.items():
                    rows.append({"section": section, "metric": key, "value": json.dumps(value)})
        pd.DataFrame(rows).to_sql("project_metrics", conn, index=False, if_exists="replace")

    return db_path


def read_table(table: str, db_path: str | Path = DB_PATH) -> pd.DataFrame:
    allowed = {"journeys", "attribution_results", "budget_baseline", "feature_importance", "project_metrics"}
    if table not in allowed:
        raise ValueError("Unknown table")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
