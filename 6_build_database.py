"""Step 6 — persist project outputs to SQLite for API/dashboard consumption."""
from src.marketing_attribution.db import build_database


if __name__ == "__main__":
    path = build_database()
    print(f"SQLite database built: {path}")
