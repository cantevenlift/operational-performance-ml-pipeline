"""Feature engineering step."""

from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

def run() -> Path:
    df = pd.read_parquet(PROCESSED / "incidents_clean.parquet")
    df["day_of_week"] = df["created_at"].dt.weekday
    df["hour"] = df["created_at"].dt.hour
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_business_hours"] = df["hour"].between(9, 17).astype(int)
    df["priority_change_interaction"] = df["in_change_window"] * df["priority"].cat.codes

    y = df["sla_breached"].astype(int)

    X = pd.get_dummies(
        df[["priority","category","assignment_group","region","channel","service",
            "day_of_week","hour","is_weekend","is_business_hours","priority_change_interaction",
            "reopened","csat_score"]],
        drop_first=False
    )

    out_x = PROCESSED / "X.parquet"
    out_y = PROCESSED / "y.parquet"
    X.to_parquet(out_x, index=False)
    y.to_frame("sla_breached").to_parquet(out_y, index=False)
    return out_x

if __name__ == "__main__":
    out = run()
    print(f"Wrote features to: {out}")
