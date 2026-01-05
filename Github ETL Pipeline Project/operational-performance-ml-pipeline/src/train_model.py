"""Model training script."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

def run() -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    X = pd.read_parquet(PROCESSED / "X.parquet")
    y = pd.read_parquet(PROCESSED / "y.parquet")["sla_breached"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, digits=3))
    print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 4))

    out = MODELS / "sla_breach_logreg.joblib"
    joblib.dump({"model": clf, "columns": list(X.columns)}, out)
    return out

if __name__ == "__main__":
    out = run()
    print(f"Saved model to: {out}")
