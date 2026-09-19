"""Sentiment classification: TF-IDF + Logistic Regression, trained from
scratch on this project's labeled data.

Real substitute for a pretrained Hugging Face transformer classifier (no
Hub access in this sandbox -- see README). Reports a genuine held-out
macro-F1, not an assumed one, because the dataset's sentiment labels were
assigned programmatically at generation time (data/generate_feedback.py).
"""
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "sentiment_model")


def train(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment_true"], test_size=0.2, random_state=42, stratify=df["sentiment_true"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    report = classification_report(y_test, preds, output_dict=True)
    macro_f1 = f1_score(y_test, preds, average="macro")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    return {"macro_f1": round(macro_f1, 4), "per_class": report}


def load():
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


def predict(texts, vectorizer=None, model=None):
    if vectorizer is None or model is None:
        vectorizer, model = load()
    X = vectorizer.transform(texts)
    return model.predict(X), model.predict_proba(X).max(axis=1)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "feedback_raw.csv"))
    metrics = train(df)
    print(f"Held-out macro-F1: {metrics['macro_f1']}")
    for label, stats in metrics["per_class"].items():
        if isinstance(stats, dict):
            print(f"  {label}: F1={stats.get('f1-score', 0):.3f} support={stats.get('support', 0)}")
