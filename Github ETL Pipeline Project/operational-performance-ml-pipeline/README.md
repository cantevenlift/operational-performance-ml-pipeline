# Operational Performance ML Pipeline

End-to-end analytics engineering + ML example: ETL, data quality checks, feature engineering, model training, and evaluation.

## Quickstart
1. Create a virtual environment
2. `pip install -r requirements.txt`
3. Run ETL: `python -m src.etl`
4. Build features: `python -m src.features`
5. Train model: `python -m src.train_model`

## Data
Synthetic but realistic sample data is provided in `data/raw/`:
- `incidents.csv`: ticket-level operational data
- `daily_kpis.csv`: daily KPI rollups

## Goal
Predict whether an incident will breach SLA (`sla_breached`) using operational features.
