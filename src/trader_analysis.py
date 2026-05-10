from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def trader_performance_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        trades_df.groupby("account")
        .agg(
            total_pnl=("closed_pnl", "sum"),
            avg_pnl=("closed_pnl", "mean"),
            median_pnl=("closed_pnl", "median"),
            trade_count=("closed_pnl", "count"),
            win_rate=("is_winning_trade", "mean"),
            avg_leverage=("leverage", "mean"),
            avg_notional=("notional_usd", "mean"),
        )
        .reset_index()
    )
    summary["win_rate"] = summary["win_rate"] * 100
    return summary.sort_values("total_pnl", ascending=False)


def market_behavior_summary(trades_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    side_perf = (
        trades_df.groupby("side")
        .agg(total_pnl=("closed_pnl", "sum"), avg_pnl=("closed_pnl", "mean"), count=("side", "size"))
        .reset_index()
        .sort_values("total_pnl", ascending=False)
    )
    symbol_activity = (
        trades_df.groupby("symbol")
        .agg(
            trades=("symbol", "count"),
            total_pnl=("closed_pnl", "sum"),
            avg_leverage=("leverage", "mean"),
        )
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    daily_activity = (
        trades_df.groupby("trade_date")
        .agg(
            trades=("trade_date", "size"),
            unique_accounts=("account", "nunique"),
            total_pnl=("closed_pnl", "sum"),
            avg_leverage=("leverage", "mean"),
        )
        .reset_index()
        .sort_values("trade_date")
    )
    return {
        "side_performance": side_perf,
        "symbol_activity": symbol_activity,
        "daily_activity": daily_activity,
    }


def sentiment_vs_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        trades_df.groupby("sentiment_classification")
        .agg(
            trades=("closed_pnl", "count"),
            total_pnl=("closed_pnl", "sum"),
            avg_pnl=("closed_pnl", "mean"),
            median_pnl=("closed_pnl", "median"),
            avg_leverage=("leverage", "mean"),
            avg_notional=("notional_usd", "mean"),
            loss_rate=("is_losing_trade", "mean"),
        )
        .reset_index()
        .sort_values("total_pnl", ascending=False)
    )
    summary["loss_rate"] = summary["loss_rate"] * 100
    return summary


def risk_exposure_by_account(trades_df: pd.DataFrame) -> pd.DataFrame:
    risk = (
        trades_df.groupby("account")
        .agg(
            max_leverage=("leverage", "max"),
            avg_leverage=("leverage", "mean"),
            avg_notional=("notional_usd", "mean"),
            pnl_std=("closed_pnl", "std"),
            largest_loss=("closed_pnl", "min"),
        )
        .reset_index()
    )
    risk["pnl_std"] = risk["pnl_std"].fillna(0.0)
    return risk.sort_values(["max_leverage", "avg_notional"], ascending=False)


def trader_clustering(trader_summary: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    required = ["total_pnl", "trade_count", "win_rate", "avg_leverage", "avg_notional"]
    available = [c for c in required if c in trader_summary.columns]
    if len(available) < 3 or len(trader_summary) < n_clusters:
        clustered = trader_summary.copy()
        clustered["cluster"] = -1
        return clustered

    features = trader_summary[available].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)

    clustered = trader_summary.copy()
    clustered["cluster"] = labels
    return clustered

