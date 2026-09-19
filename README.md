# NLP-Driven Customer Feedback Analysis Dashboard

Processes customer reviews/support tickets end to end — sentiment
classification, NER-based preprocessing, unsupervised topic discovery — and
surfaces the result in an interactive Streamlit dashboard with sentiment /
topic / segment / time-period filters.

## Honest scope note (read this first)

Same sandbox constraint as the other three projects: **no outbound access
to Hugging Face Hub** (only PyPI and GitHub are reachable). That changes
which concrete models back two of the four NLP stages:

| Stage | Resume said | This build uses | Why |
|---|---|---|---|
| NER / linguistic preprocessing | spaCy | spaCy `en_core_web_sm` — **installed from its GitHub release wheel**, not the Hub, so this is the *actual* pretrained model, real NER, no substitution needed | GitHub is reachable, HF Hub is not |
| Sentiment classification | Hugging Face transformer (0.88 F1) | `nlp/sentiment.py`: TF-IDF + Logistic Regression, trained from scratch on this project's labeled data | No pretrained sentiment transformer reachable; this is a real trained classifier with a held-out F1 reported below, not a stub |
| Topic modeling | BERTopic (sentence-transformer embeddings + UMAP + HDBSCAN) | `nlp/topics.py`: NMF over TF-IDF (scikit-learn) | BERTopic's default embedding backend is a Hub-hosted sentence-transformer; NMF/TF-IDF is the standard classical alternative and needs nothing external |
| Dataset | 25K+ real customer reviews/tickets | ~2,400 synthetic but structurally realistic reviews (`data/generate_feedback.py`) across 7 issue categories, 3 customer segments, 12 months, with programmatically-assigned ground-truth sentiment/category labels | No real customer data available in a sandbox; ground truth labels also let this report a genuine held-out F1 instead of an assumed one |

The pipeline architecture, the Streamlit dashboard, and the evaluation
methodology are all real; swapping in a Hub-hosted sentiment model or
BERTopic later is a two-function change (`nlp/sentiment.py::predict`,
`nlp/topics.py::assign_topics`).

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
