"""Runs sentiment classification + NER preprocessing + topic modeling over
the full feedback dataset and writes the enriched CSV the dashboard reads,
plus a metrics.json summarizing each stage."""
import json
import os
import time

import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nlp import ner_preprocessing, sentiment, topics

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feedback_raw.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
ENRICHED_PATH = os.path.join(OUT_DIR, "feedback_enriched.csv")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} feedback rows")

    t0 = time.time()
    sentiment_metrics = sentiment.train(df)
    vectorizer, model = sentiment.load()
    preds, confidences = sentiment.predict(df["text"], vectorizer, model)
    df["sentiment_pred"] = preds
    df["sentiment_confidence"] = confidences
    print(f"Sentiment: macro-F1={sentiment_metrics['macro_f1']} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    df = ner_preprocessing.run(df)
    manual_review_reduction = (df["category_inferred_from_keyphrase"] == df["category_true"]).mean()
    print(f"NER/preprocessing done ({time.time()-t0:.1f}s), "
          f"key-phrase-only inference accuracy={manual_review_reduction:.1%}")

    t0 = time.time()
    assignments, topic_terms, _, _ = topics.fit_topics(df["text"].tolist())
    df["topic_id"] = assignments
    topic_metrics = topics.evaluate_against_ground_truth(assignments, df["category_true"])
    print(f"Topics: NMI={topic_metrics['nmi']} ({time.time()-t0:.1f}s)")

    df.to_csv(ENRICHED_PATH, index=False)
    print(f"Wrote enriched dataset -> {ENRICHED_PATH}")

    metrics = {
        "n_reviews": len(df),
        "sentiment_classifier": {
            "macro_f1": sentiment_metrics["macro_f1"],
            "per_class_f1": {
                k: round(v["f1-score"], 4) for k, v in sentiment_metrics["per_class"].items()
                if isinstance(v, dict) and "f1-score" in v
            },
        },
        "ner_preprocessing": {
            "key_phrase_only_inference_accuracy": round(float(manual_review_reduction), 4),
            "entity_type_counts": df["entity_types"].str.split(",").explode().value_counts().to_dict(),
        },
        "topic_model": {
            "n_topics": len(topic_terms),
            "nmi_vs_ground_truth_category": topic_metrics["nmi"],
            "topic_terms": {str(k): v for k, v in topic_terms.items()},
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
