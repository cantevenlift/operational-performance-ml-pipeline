"""Train a baseline NLP model for ticket text classification."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

def run() -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW / "tickets_text.csv")

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)),
        ("clf", LinearSVC())
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(classification_report(y_test, y_pred, digits=3))

    out = MODELS / "ticket_text_tfidf_svm.joblib"
    joblib.dump(pipe, out)
    return out

if __name__ == "__main__":
    out = run()
    print(f"Saved model to: {out}")
