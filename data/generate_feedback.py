"""Generates a synthetic-but-structurally-realistic customer feedback
dataset, scaled down from the resume's "25K+ customer reviews and support
tickets" to ~2,400 rows (see top-level README's honest-scope section).

Each row gets a programmatically-assigned ground-truth sentiment and issue
category, which is what lets nlp/sentiment.py and nlp/topics.py report a
genuine held-out accuracy/NMI instead of an assumed one.
"""
import csv
import os
import random

random.seed(42)

OUT_PATH = os.path.join(os.path.dirname(__file__), "feedback_raw.csv")

COMPANIES = [
    "Initech", "Globex Inc", "Acme Corp", "Umbrella Labs", "Stark Data Systems",
    "Wayne Analytics", "Hooli", "Massive Dynamic", "Soylent Corp", "Cyberdyne Systems",
]
PRODUCTS = ["DataSync Pro", "QuickReport", "InsightBoard", "FlowPilot", "MetricHub"]
SEGMENTS = ["Enterprise", "SMB", "Startup"]

CATEGORIES = {
    "billing": {
        "sentiment": "negative",
        "templates": [
            "We were double-charged for {product} this month and support at {company} has been slow to respond.",
            "The invoice for {product} doesn't match our contract pricing -- billing needs to fix this ASAP.",
            "Getting a refund for the {product} overcharge took three weeks. Very frustrating billing process.",
        ],
    },
    "performance": {
        "sentiment": "negative",
        "templates": [
            "{product} has been timing out constantly since the last update, our team at {company} can't get work done.",
            "Dashboard loads take over 30 seconds in {product}, this is unacceptable for {company}'s reporting needs.",
            "We're seeing frequent 500 errors in {product} during peak hours.",
        ],
    },
    "support": {
        "sentiment": "negative",
        "templates": [
            "Opened a ticket about {product} five days ago and still no response from support.",
            "The support team at {company} keeps closing our {product} tickets without actually resolving the issue.",
            "Support wait times for {product} have gotten noticeably worse this quarter.",
        ],
    },
    "bug": {
        "sentiment": "negative",
        "templates": [
            "Found a bug in {product} where exported reports show the wrong currency symbol.",
            "{product}'s new filter feature crashes the app when we select more than 5 tags.",
            "Data sync between {product} and our CRM has been dropping records intermittently.",
        ],
    },
    "onboarding": {
        "sentiment": "neutral",
        "templates": [
            "The onboarding docs for {product} are a bit outdated but our team at {company} figured it out eventually.",
            "Setup for {product} took longer than expected, would be nice to have a clearer setup wizard.",
            "New team members at {company} find the {product} admin console confusing at first.",
        ],
    },
    "feature_request": {
        "sentiment": "neutral",
        "templates": [
            "Would love to see a dark mode added to {product} in a future release.",
            "It would help {company} a lot if {product} supported bulk export to Excel.",
            "Any plans to add SSO support to {product}? We'd need that to expand usage at {company}.",
        ],
    },
    "praise": {
        "sentiment": "positive",
        "templates": [
            "{product} has completely streamlined reporting for {company}, couldn't be happier with the results.",
            "Support resolved our {product} issue within the hour -- great experience overall.",
            "The new {product} update is fantastic, our team at {company} loves the redesigned interface.",
            "{product} just works. Rock solid uptime and the {company} team is very satisfied.",
        ],
    },
}

MONTHS_2025_2026 = [f"2025-{m:02d}" for m in range(9, 13)] + [f"2026-{m:02d}" for m in range(1, 10)]


def make_row(row_id):
    category = random.choice(list(CATEGORIES.keys()))
    cat_info = CATEGORIES[category]
    template = random.choice(cat_info["templates"])
    company = random.choice(COMPANIES)
    product = random.choice(PRODUCTS)
    text = template.format(product=product, company=company)

    segment = random.choice(SEGMENTS)
    month = random.choice(MONTHS_2025_2026)
    day = random.randint(1, 28)

    return {
        "review_id": row_id,
        "date": f"{month}-{day:02d}",
        "company": company,
        "segment": segment,
        "product": product,
        "category_true": category,
        "sentiment_true": cat_info["sentiment"],
        "text": text,
    }


def add_label_noise(rows, noise_rate=0.10):
    """Real-world sentiment labels are noisy (human annotators / implied
    tone disagree ~10-15% of the time even on product review datasets).
    Flipping a slice of labels to a neighboring class keeps the sentiment
    classification task honest instead of trivially separable by template
    n-grams, which would inflate F1 to an unrealistic 1.0."""
    neighbor = {"negative": "neutral", "neutral": random.choice(["negative", "positive"]), "positive": "neutral"}
    n_flip = int(len(rows) * noise_rate)
    idxs = random.sample(range(len(rows)), n_flip)
    for i in idxs:
        true_label = rows[i]["sentiment_true"]
        rows[i]["sentiment_true"] = neighbor.get(true_label, true_label)
    return rows


def main(n=2400):
    rows = [make_row(i) for i in range(1, n + 1)]
    rows = add_label_noise(rows, noise_rate=0.10)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic feedback rows -> {OUT_PATH}")
    from collections import Counter
    print("Sentiment distribution:", Counter(r["sentiment_true"] for r in rows))
    print("Category distribution:", Counter(r["category_true"] for r in rows))


if __name__ == "__main__":
    main()
