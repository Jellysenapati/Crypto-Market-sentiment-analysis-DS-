from __future__ import annotations

from pathlib import Path
import json
import logging
import re
from typing import Iterable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
CHARTS_DIR = ROOT_DIR / "outputs" / "charts"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
INSIGHTS_DIR = ROOT_DIR / "outputs" / "insights"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def parse_datetime_column(
    series: pd.Series,
    utc: bool = True,
    dayfirst: bool = False,
) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=utc, dayfirst=dayfirst)

