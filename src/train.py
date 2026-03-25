from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an intent router model (TF-IDF + LogisticRegression)."
    )
    p.add_argument("--data", type=Path, default=Path("data/train.csv"))
    p.add_argument("--model-out", type=Path, default=Path("artifacts/intent_router.joblib"))
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("train.csv must contain columns: text,label")

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    # Stratify only works if each class has >= 2 samples in each split.
    # With tiny datasets, stratify can fail; we handle that gracefully.
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, args.model_out)
    print(f"\nSaved model to {args.model_out}")


if __name__ == "__main__":
    main()
