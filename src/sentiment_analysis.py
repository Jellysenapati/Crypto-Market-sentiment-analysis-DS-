from __future__ import annotations

import pandas as pd


def sentiment_distribution(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        sentiment_df["sentiment_classification"]
        .value_counts(dropna=False)
        .rename_axis("sentiment_classification")
        .reset_index(name="count")
    )
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100
    return counts


def fear_vs_greed_frequency(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Extreme Fear": "Fear",
        "Fear": "Fear",
        "Neutral": "Neutral",
        "Greed": "Greed",
        "Extreme Greed": "Greed",
    }
    grouped = sentiment_df.copy()
    grouped["sentiment_bucket"] = grouped["sentiment_classification"].map(mapping).fillna("Other")
    return (
        grouped["sentiment_bucket"]
        .value_counts()
        .rename_axis("sentiment_bucket")
        .reset_index(name="count")
    )


def sentiment_timeline(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    timeline = (
        sentiment_df.groupby(["date", "sentiment_classification"])
        .size()
        .rename("observations")
        .reset_index()
        .sort_values("date")
    )
    return timeline


def extreme_sentiment_occurrences(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    mask = sentiment_df["sentiment_classification"].isin(["Extreme Fear", "Extreme Greed"])
    return sentiment_df.loc[mask].copy().sort_values("date")

