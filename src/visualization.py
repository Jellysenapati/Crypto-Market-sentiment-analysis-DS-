from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def _save_matplotlib_chart(filename: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_sentiment_distribution(distribution_df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=distribution_df, x="sentiment_classification", y="count", palette="viridis")
    plt.title("Market Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=25)
    return _save_matplotlib_chart("sentiment_distribution.png", output_dir)


def plot_sentiment_timeline(timeline_df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=timeline_df,
        x="date",
        y="observations",
        hue="sentiment_classification",
        linewidth=2,
    )
    plt.title("Sentiment Timeline Trend")
    plt.xlabel("Date")
    plt.ylabel("Observations")
    return _save_matplotlib_chart("sentiment_timeline.png", output_dir)


def plot_pnl_distribution(trades_df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(10, 6))
    sns.histplot(trades_df["closed_pnl"], bins=50, kde=True, color="#4c72b0")
    plt.title("PnL Distribution")
    plt.xlabel("Closed PnL")
    plt.ylabel("Frequency")
    return _save_matplotlib_chart("pnl_distribution.png", output_dir)


def plot_leverage_vs_pnl(trades_df: pd.DataFrame, output_dir: Path) -> Path:
    sample = trades_df.sample(min(len(trades_df), 5000), random_state=42) if len(trades_df) else trades_df
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sample,
        x="leverage",
        y="closed_pnl",
        hue="sentiment_classification",
        alpha=0.6,
    )
    plt.title("Leverage vs PnL by Sentiment")
    plt.xlabel("Leverage")
    plt.ylabel("Closed PnL")
    return _save_matplotlib_chart("leverage_vs_pnl.png", output_dir)


def plot_sentiment_boxplot(trades_df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=trades_df, x="sentiment_classification", y="closed_pnl", palette="Set2")
    plt.title("PnL by Sentiment Regime")
    plt.xlabel("Sentiment")
    plt.ylabel("Closed PnL")
    plt.xticks(rotation=25)
    return _save_matplotlib_chart("pnl_boxplot_by_sentiment.png", output_dir)


def plot_correlation_heatmap(trades_df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(10, 8))
    numeric = trades_df[["execution_price", "size", "start_position", "closed_pnl", "leverage", "notional_usd"]]
    corr = numeric.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    return _save_matplotlib_chart("correlation_heatmap.png", output_dir)


def plot_buy_sell_pie(side_performance_df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(8, 8))
    plt.pie(
        side_performance_df["count"],
        labels=side_performance_df["side"],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Buy vs Sell Trade Share")
    return _save_matplotlib_chart("buy_sell_pie.png", output_dir)


def plotly_interactive_sentiment_pnl(trades_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = px.box(
        trades_df,
        x="sentiment_classification",
        y="closed_pnl",
        color="sentiment_classification",
        title="Interactive PnL Distribution by Sentiment",
        points="outliers",
    )
    path = output_dir / "interactive_sentiment_pnl.html"
    fig.write_html(path)
    return path

