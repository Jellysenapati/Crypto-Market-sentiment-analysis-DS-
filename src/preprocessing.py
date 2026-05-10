from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .utils import normalize_columns, parse_datetime_column

LOGGER = logging.getLogger(__name__)


def clean_sentiment_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = df.rename(columns={"classification": "sentiment_classification"})
    df["date"] = parse_datetime_column(df["date"], utc=True).dt.floor("D")
    df["sentiment_classification"] = (
        df["sentiment_classification"].astype(str).str.strip().str.title()
    )

    df = df.dropna(subset=["date", "sentiment_classification"]).drop_duplicates()
    df = df[df["sentiment_classification"].ne("")]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_trader_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    rename_map = {
        "execution_price": "execution_price",
        "closed_pnl": "closed_pnl",
        "start_position": "start_position",
    }
    df = df.rename(columns=rename_map)

    df["time"] = parse_datetime_column(df["time"], utc=True)
    numeric_cols = ["execution_price", "size", "start_position", "closed_pnl", "leverage"]
    df = _coerce_numeric(df, numeric_cols)

    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["account"] = df["account"].astype(str).str.strip()
    df["event"] = df["event"].astype(str).str.strip()

    before = len(df)
    df = df.drop_duplicates()
    deduped = before - len(df)
    if deduped:
        LOGGER.info("Dropped %s duplicate trader rows.", deduped)

    df = df.dropna(subset=["time", "account", "symbol", "closed_pnl"])
    df = df[df["execution_price"].gt(0) | df["execution_price"].isna()]
    df = df[df["size"].gt(0) | df["size"].isna()]
    df = df[df["leverage"].fillna(0).ge(0)]
    df["trade_date"] = df["time"].dt.floor("D")
    df["notional_usd"] = (df["execution_price"] * df["size"]).abs()
    df["is_winning_trade"] = df["closed_pnl"] > 0
    df["is_losing_trade"] = df["closed_pnl"] < 0

    for col in ["execution_price", "size", "start_position", "leverage"]:
        median_val = df[col].median(skipna=True)
        if np.isfinite(median_val):
            df[col] = df[col].fillna(median_val)

    df = df.sort_values("time").reset_index(drop=True)
    return df


def merge_sentiment_and_trades(
    sentiment_df: pd.DataFrame, trader_df: pd.DataFrame
) -> pd.DataFrame:
    sentiment = sentiment_df.copy()
    sentiment["trade_date"] = sentiment["date"]

    merged = trader_df.merge(
        sentiment[["trade_date", "sentiment_classification"]],
        on="trade_date",
        how="left",
    )
    merged["sentiment_classification"] = merged["sentiment_classification"].fillna("Unknown")
    return merged

