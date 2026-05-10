from __future__ import annotations

from pathlib import Path
import logging
from typing import Callable, Sequence

import pandas as pd

from .utils import RAW_DATA_DIR, normalize_columns

LOGGER = logging.getLogger(__name__)

REQUIRED_SENTIMENT_COLUMNS = {"date", "classification"}
REQUIRED_TRADER_COLUMNS = {
    "account",
    "symbol",
    "execution_price",
    "size",
    "side",
    "time",
    "start_position",
    "event",
    "closed_pnl",
    "leverage",
}

TRADER_COLUMN_ALIASES: dict[str, list[str]] = {
    "account": ["account"],
    "symbol": ["symbol", "coin", "asset", "market"],
    "execution_price": ["execution_price", "price", "entry_price"],
    "size": ["size", "size_tokens", "qty", "quantity", "position_size", "size_usd"],
    "side": ["side", "direction"],
    "time": ["time", "timestamp", "created_at", "trade_time", "timestamp_ist"],
    "start_position": ["start_position", "startposition", "position", "position_size"],
    "event": ["event", "type", "trade_type"],
    "closed_pnl": ["closed_pnl", "pnl", "realized_pnl", "profit_loss"],
    "leverage": ["leverage", "lev", "margin_leverage"],
}


class SchemaValidationError(ValueError):
    """Raised when dataset schema does not match required fields."""


def _read_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. Place files inside data/raw/."
        )

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    raise ValueError(
        f"Unsupported file extension '{suffix}' for file '{file_path.name}'."
    )


def find_dataset_path(filename_candidates: Sequence[str]) -> Path:
    for name in filename_candidates:
        candidate = RAW_DATA_DIR / name
        if candidate.exists():
            return candidate
    expected = ", ".join(filename_candidates)
    raise FileNotFoundError(
        f"No dataset found in {RAW_DATA_DIR}. Tried: {expected}. "
        "Please add your downloaded files and rename to one of these options."
    )


def _find_by_schema(
    required_columns: set[str],
    dataset_name: str,
    pre_validate_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> Path:
    supported_files = sorted(
        [
            path
            for path in RAW_DATA_DIR.glob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls", ".parquet"}
        ]
    )
    if not supported_files:
        raise FileNotFoundError(
            f"No supported files found in {RAW_DATA_DIR}. "
            "Add CSV/XLSX/XLS/PARQUET files to data/raw/."
        )

    schema_failures: list[str] = []
    for candidate in supported_files:
        try:
            df = _read_dataset(candidate)
            if pre_validate_transform is not None:
                df = pre_validate_transform(df)
            validate_schema(df, required_columns, dataset_name)
            LOGGER.info("Auto-detected %s file by schema: %s", dataset_name, candidate.name)
            return candidate
        except Exception as exc:  # pragma: no cover - defensive path
            schema_failures.append(f"{candidate.name}: {exc}")

    raise FileNotFoundError(
        f"Could not find a valid {dataset_name} file by schema in {RAW_DATA_DIR}.\n"
        "Checked files:\n- " + "\n- ".join(schema_failures)
    )


def validate_schema(df: pd.DataFrame, required_columns: set[str], dataset_name: str) -> None:
    normalized_df = normalize_columns(df)
    available_columns = set(normalized_df.columns)
    missing = sorted(required_columns - available_columns)
    if missing:
        available = ", ".join(sorted(available_columns))
        missing_text = ", ".join(missing)
        raise SchemaValidationError(
            f"{dataset_name} schema validation failed.\n"
            f"Missing required columns: {missing_text}\n"
            f"Available columns: {available}"
        )


def harmonize_trader_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_columns(df)
    rename_map: dict[str, str] = {}
    existing = set(normalized.columns)

    for canonical, aliases in TRADER_COLUMN_ALIASES.items():
        if canonical in existing:
            continue
        for alias in aliases:
            if alias in existing:
                rename_map[alias] = canonical
                break

    normalized = normalized.rename(columns=rename_map)

    # Create safe defaults when unavailable in source export.
    if "event" not in normalized.columns:
        normalized["event"] = "unknown"
    if "leverage" not in normalized.columns:
        normalized["leverage"] = 1.0

    return normalized


def load_sentiment_data(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        try:
            path = find_dataset_path(
                [
                    "fear_greed.csv",
                    "fear_and_greed.csv",
                    "fear_greed_index.csv",
                    "fear_greed.xlsx",
                ]
            )
        except FileNotFoundError:
            path = _find_by_schema(REQUIRED_SENTIMENT_COLUMNS, "Fear & Greed dataset")
    df = _read_dataset(path)
    validate_schema(df, REQUIRED_SENTIMENT_COLUMNS, "Fear & Greed dataset")
    LOGGER.info("Loaded sentiment data: %s rows from %s", len(df), path.name)
    return df


def load_trader_data(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        try:
            path = find_dataset_path(
                [
                    "historical_trader_data.csv",
                    "trader_data.csv",
                    "hyperliquid_trades.csv",
                    "historical_trader_data.xlsx",
                ]
            )
        except FileNotFoundError:
            path = _find_by_schema(
                REQUIRED_TRADER_COLUMNS,
                "Historical trader dataset",
                pre_validate_transform=harmonize_trader_columns,
            )
    df = _read_dataset(path)
    df = harmonize_trader_columns(df)
    validate_schema(df, REQUIRED_TRADER_COLUMNS, "Historical trader dataset")
    LOGGER.info("Loaded trader data: %s rows from %s", len(df), path.name)
    return df

