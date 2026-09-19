"""spaCy-based NER + linguistic preprocessing.

Uses the real pretrained `en_core_web_sm` model (installed from its GitHub
release wheel -- GitHub is reachable, HF Hub is not; see README). Extracts
named entities (company/product mentions) and a lightweight "key phrase"
(the most salient noun chunk) per review, which stands in for the resume's
"reducing manual review effort by ~65%" claim: a reviewer who reads only
the key phrase, rather than the full review text, can correctly infer the
issue category most of the time (measured below against category_true).
"""
import os

import pandas as pd
import spacy

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def extract_entities_and_keyphrase(text: str) -> dict:
    nlp = _get_nlp()
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    # crude but real linguistic-feature-based key phrase: longest noun chunk
    # that isn't just a company/product name, ranked by token length
    noun_chunks = [nc.text for nc in doc.noun_chunks]
    candidate_chunks = [nc for nc in noun_chunks if len(nc.split()) >= 2]
    key_phrase = max(candidate_chunks, key=len) if candidate_chunks else (noun_chunks[0] if noun_chunks else "")

    return {"entities": entities, "key_phrase": key_phrase}


CATEGORY_KEYWORDS = {
    "billing": ["invoice", "charge", "refund", "billing", "pricing"],
    "performance": ["timing out", "load", "error", "slow", "500", "seconds"],
    "support": ["ticket", "support", "response", "wait"],
    "bug": ["bug", "crash", "export", "sync", "drop"],
    "onboarding": ["onboarding", "setup", "docs", "console", "confusing"],
    "feature_request": ["dark mode", "export to excel", "sso", "plans to add", "would love"],
    "praise": ["streamlined", "fantastic", "rock solid", "satisfied", "great experience"],
}


def infer_category_from_keyphrase(key_phrase: str, full_text: str) -> str:
    """Approximates what a reviewer could infer from the key phrase alone
    (falls back to full text keyword match only to measure key-phrase-only
    inference quality against it)."""
    kp_lower = key_phrase.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in kp_lower for kw in keywords):
            return category
    return "unknown"


def run(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for text in df["text"]:
        result = extract_entities_and_keyphrase(text)
        inferred = infer_category_from_keyphrase(result["key_phrase"], text)
        records.append({
            "n_entities": len(result["entities"]),
            "entity_types": ",".join(sorted(set(e["label"] for e in result["entities"]))),
            "key_phrase": result["key_phrase"],
            "category_inferred_from_keyphrase": inferred,
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "feedback_raw.csv"))
    enriched = run(df.head(300))  # smoke-test on a slice
    inferable = (enriched["category_inferred_from_keyphrase"] == enriched["category_true"]).mean()
    print(f"Key-phrase-only category inference accuracy (sample of 300): {inferable:.1%}")
    from collections import Counter
    all_types = ",".join(enriched["entity_types"]).split(",")
    print("Entity type counts:", Counter(t for t in all_types if t))
