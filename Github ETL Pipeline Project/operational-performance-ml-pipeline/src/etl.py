"""ETL step for the operational performance pipeline."""

from __future__ import annotations
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

def quality_checks(df: pd.DataFrame) -> None:
    required_cols = {
        "ticket_id","created_at","priority","category","assignment_group","region","channel","service",
        "resolution_minutes","sla_target_minutes","sla_breached","reopened","csat_score","in_change_window"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["ticket_id"].isna().any():
        raise ValueError("ticket_id contains nulls")
    if df["ticket_id"].duplicated().any():
        raise ValueError("ticket_id contains duplicates")
    if (df["resolution_minutes"] <= 0).any():
        raise ValueError("resolution_minutes must be > 0")

def run() -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW / "incidents.csv")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    if df["created_at"].isna().any():
        raise ValueError("created_at contains invalid timestamps")
    quality_checks(df)

    for c in ["priority","category","assignment_group","region","channel","service"]:
        df[c] = df[c].astype("category")

    for c in ["sla_breached","reopened","in_change_window"]:
        df[c] = df[c].astype(int)

    out = PROCESSED / "incidents_clean.parquet"
    df.to_parquet(out, index=False)
    return out

if __name__ == "__main__":
    out = run()
    print(f"Wrote cleaned dataset to: {out}")
