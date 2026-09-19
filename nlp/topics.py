"""Unsupervised topic modeling via NMF over TF-IDF.

Real substitute for BERTopic (whose default embedding backend is a
Hub-hosted sentence-transformer, unreachable in this sandbox -- see
README). NMF/TF-IDF is the standard classical alternative and needs no
external model. Evaluated against the dataset's ground-truth issue
categories via normalized mutual information (NMI) -- an honest measure of
how well unsupervised topics line up with real structure, since a perfect
score isn't expected or meaningful here.
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score

N_TOPICS = 8
N_TOP_WORDS = 8

# Product/company names appear in nearly every synthetic review (every
# template mentions one), so without excluding them they dominate every
# NMF component as generic high-frequency terms and prevent the model from
# separating on actual issue content. A real customer-feedback corpus has
# the same problem with the company's own product name -- standard practice
# is to add such near-universal terms to the stopword list before topic
# modeling, which is what this does.
DOMAIN_STOPWORDS = frozenset({
    "datasync", "pro", "quickreport", "insightboard", "flowpilot", "metrichub",
    "initech", "globex", "inc", "acme", "corp", "umbrella", "labs", "stark",
    "systems", "wayne", "analytics", "hooli", "massive", "dynamic", "soylent",
    "cyberdyne", "team", "company",
})
CUSTOM_STOPWORDS = list(ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS))


def fit_topics(texts, n_topics=N_TOPICS):
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=3, stop_words=CUSTOM_STOPWORDS, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(texts)

    nmf = NMF(n_components=n_topics, random_state=42, max_iter=400)
    doc_topic = nmf.fit_transform(tfidf)
    topic_assignments = doc_topic.argmax(axis=1)

    feature_names = np.array(vectorizer.get_feature_names_out())
    topic_terms = {}
    for i, component in enumerate(nmf.components_):
        top_idx = component.argsort()[::-1][:N_TOP_WORDS]
        topic_terms[i] = feature_names[top_idx].tolist()

    return topic_assignments, topic_terms, vectorizer, nmf


def evaluate_against_ground_truth(topic_assignments, category_true):
    nmi = normalized_mutual_info_score(category_true, topic_assignments)
    return {"nmi": round(nmi, 4)}


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "feedback_raw.csv"))
    assignments, terms, _, _ = fit_topics(df["text"].tolist())
    metrics = evaluate_against_ground_truth(assignments, df["category_true"])

    print(f"NMI vs ground-truth categories: {metrics['nmi']}")
    print("\nDiscovered topics (top terms):")
    for i, words in terms.items():
        n_docs = int((assignments == i).sum())
        print(f"  Topic {i} ({n_docs} reviews): {', '.join(words)}")
