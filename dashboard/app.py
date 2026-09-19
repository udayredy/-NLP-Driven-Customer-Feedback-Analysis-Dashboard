"""Streamlit dashboard for the enriched customer feedback dataset.

Filters: sentiment, topic, customer segment, time period -- mirrors the
resume's "interactive Streamlit dashboard with sentiment, topic,
customer-segment, and time-period filters" line exactly.
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

ENRICHED_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "feedback_enriched.csv")

st.set_page_config(page_title="Customer Feedback Analytics", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(ENRICHED_PATH, parse_dates=["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


if not os.path.exists(ENRICHED_PATH):
    st.error(
        "No enriched dataset found. Run `python data/generate_feedback.py` "
        "then `python scripts/build_pipeline.py` first."
    )
    st.stop()

df = load_data()

st.title("NLP-Driven Customer Feedback Analysis Dashboard")
st.caption(
    "Sentiment: TF-IDF + Logistic Regression · NER/preprocessing: spaCy `en_core_web_sm` · "
    "Topics: NMF over TF-IDF (see README for why these substitute for a Hub-hosted "
    "transformer + BERTopic in this sandbox)"
)

with st.sidebar:
    st.header("Filters")
    sentiments = st.multiselect("Sentiment", sorted(df["sentiment_pred"].unique()), default=list(df["sentiment_pred"].unique()))
    segments = st.multiselect("Customer segment", sorted(df["segment"].unique()), default=list(df["segment"].unique()))
    topic_options = sorted(df["topic_id"].unique())
    topics_selected = st.multiselect("Topic ID", topic_options, default=list(topic_options))
    months = sorted(df["month"].unique())
    month_range = st.select_slider("Time period", options=months, value=(months[0], months[-1]))

mask = (
    df["sentiment_pred"].isin(sentiments)
    & df["segment"].isin(segments)
    & df["topic_id"].isin(topics_selected)
    & (df["month"] >= month_range[0])
    & (df["month"] <= month_range[1])
)
filtered = df[mask]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Reviews (filtered)", f"{len(filtered):,}", f"of {len(df):,} total")
neg_rate = (filtered["sentiment_pred"] == "negative").mean() if len(filtered) else 0
col2.metric("% Negative", f"{neg_rate:.1%}")
col3.metric("Avg. entities/review", f"{filtered['n_entities'].mean():.1f}" if len(filtered) else "-")
col4.metric("Distinct topics shown", filtered["topic_id"].nunique())

tab1, tab2, tab3, tab4 = st.tabs(["Sentiment over time", "Topics", "Segments", "Flagged reviews"])

with tab1:
    trend = filtered.groupby(["month", "sentiment_pred"]).size().reset_index(name="count")
    fig = px.area(trend, x="month", y="count", color="sentiment_pred", title="Sentiment volume by month")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    topic_counts = filtered["topic_id"].value_counts().sort_index().reset_index()
    topic_counts.columns = ["topic_id", "count"]
    fig2 = px.bar(topic_counts, x="topic_id", y="count", title="Reviews per discovered topic")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Topic IDs come from unsupervised NMF over TF-IDF (no fixed labels) -- "
        "see outputs/metrics.json -> topic_model.topic_terms for each topic's top terms, "
        "and the README for the NMI-based evaluation against ground-truth categories."
    )

with tab3:
    seg_sent = filtered.groupby(["segment", "sentiment_pred"]).size().reset_index(name="count")
    fig3 = px.bar(seg_sent, x="segment", y="count", color="sentiment_pred", barmode="group", title="Sentiment by customer segment")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("Negative, low-confidence, or otherwise flagged reviews")
    flagged = filtered[(filtered["sentiment_pred"] == "negative")].sort_values("sentiment_confidence", ascending=False)
    st.dataframe(
        flagged[["date", "company", "segment", "product", "sentiment_pred", "sentiment_confidence", "topic_id", "key_phrase", "text"]],
        use_container_width=True,
        height=400,
    )
