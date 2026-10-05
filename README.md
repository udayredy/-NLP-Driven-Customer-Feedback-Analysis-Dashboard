# NLP-Driven Customer Feedback Analysis Dashboard

Processes customer reviews/support tickets end to end — sentiment
classification, NER-based preprocessing, unsupervised topic discovery — and
surfaces the result in an interactive Streamlit dashboard with sentiment /
topic / segment / time-period filters.

## Pipeline

```
data/generate_feedback.py  -> synthetic labeled review corpus (data/feedback_raw.csv)
nlp/sentiment.py           -> TF-IDF + LogisticRegression sentiment classifier (train + eval)
nlp/ner_preprocessing.py   -> spaCy NER + noun-phrase key-term extraction, per review
nlp/topics.py              -> NMF topic model (8 topics) fit over TF-IDF, per-review topic assignment
scripts/build_pipeline.py  -> runs all of the above, writes data/feedback_enriched.csv
dashboard/app.py           -> Streamlit dashboard reading feedback_enriched.csv
```

## Run it

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # or install the GitHub release wheel directly if HF/pip's spaCy downloader is blocked
python data/generate_feedback.py
python scripts/build_pipeline.py
streamlit run dashboard/app.py
```

## Results (this sandbox, 2,400 synthetic reviews)

See `outputs/metrics.json` after running `scripts/build_pipeline.py`:

- **Sentiment classifier**: held-out macro-F1 and per-class F1 against the
  programmatically-assigned ground truth (see `nlp/sentiment.py` output).
- **NER/preprocessing**: entity counts by type, and the fraction of reviews
  whose top-line issue is inferable from key phrases alone without reading
  the full text (this project's stand-in for "reduced manual review effort
  by ~65%").
- **Topic model**: 8 NMF topics with top terms, plus normalized mutual
  information (NMI) between discovered topics and the 7 ground-truth issue
  categories, as an honest measure of how well the unsupervised topics line
  up with real underlying structure.

## Repo layout
```
data/generate_feedback.py     # synthetic labeled dataset generator
nlp/sentiment.py                # TF-IDF + LogisticRegression classifier
nlp/ner_preprocessing.py        # spaCy NER + key-phrase extraction
nlp/topics.py                   # NMF topic modeling
scripts/build_pipeline.py       # orchestrates the above, writes enriched CSV + metrics
dashboard/app.py                # Streamlit dashboard
outputs/                        # metrics.json, feedback_enriched.csv (produced by running it)
```
