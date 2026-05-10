from __future__ import annotations

from pathlib import Path
import pandas as pd

from .utils import save_text


def _safe_row(df: pd.DataFrame, sentiment: str) -> pd.Series | None:
    subset = df[df["sentiment_classification"] == sentiment]
    if subset.empty:
        return None
    return subset.iloc[0]


def generate_insights_text(
    sentiment_perf_df: pd.DataFrame,
    trader_summary_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> str:
    greed_row = _safe_row(sentiment_perf_df, "Greed")
    fear_row = _safe_row(sentiment_perf_df, "Fear")
    extreme_greed_row = _safe_row(sentiment_perf_df, "Extreme Greed")
    extreme_fear_row = _safe_row(sentiment_perf_df, "Extreme Fear")

    top_profitable = trader_summary_df.head(5)
    highest_risk = risk_df.sort_values("max_leverage", ascending=False).head(5)
    most_consistent = trader_summary_df.sort_values(
        ["win_rate", "avg_pnl"], ascending=[False, False]
    ).head(5)

    lines = [
        "# Crypto Market Sentiment Insights Report",
        "",
        "## Executive Findings",
    ]

    if greed_row is not None and fear_row is not None:
        lines.append(
            f"- Traders are {'more' if greed_row['avg_pnl'] > fear_row['avg_pnl'] else 'less'} "
            f"profitable during Greed (avg pnl {greed_row['avg_pnl']:.2f}) vs Fear "
            f"(avg pnl {fear_row['avg_pnl']:.2f})."
        )
        lines.append(
            f"- Loss rate is {fear_row['loss_rate']:.2f}% in Fear compared with "
            f"{greed_row['loss_rate']:.2f}% in Greed."
        )

    if extreme_greed_row is not None:
        lines.append(
            f"- Average leverage during Extreme Greed is {extreme_greed_row['avg_leverage']:.2f}, "
            "indicating whether risk appetite increases in bullish periods."
        )
    if extreme_fear_row is not None:
        lines.append(
            f"- Extreme Fear has average leverage {extreme_fear_row['avg_leverage']:.2f} "
            f"and average pnl {extreme_fear_row['avg_pnl']:.2f}."
        )

    lines.extend(
        [
            "",
            "## Top Profitable Traders",
            top_profitable.to_markdown(index=False),
            "",
            "## Highest Risk Exposure Accounts",
            highest_risk.to_markdown(index=False),
            "",
            "## Most Consistent Traders",
            most_consistent.to_markdown(index=False),
            "",
            "## Actionable Strategy Suggestions",
            "- Reduce leverage during Fear and Extreme Fear regimes when loss rates rise.",
            "- Use sentiment-aware risk caps (max leverage and max notional) by market regime.",
            "- Prioritize symbols and sides with stable win rates rather than highest single-trade returns.",
            "- Monitor consistency metrics (win rate + pnl stability) for account evaluation.",
        ]
    )
    return "\n".join(lines)


def write_insights_report(content: str, output_path: Path) -> None:
    save_text(content, output_path)

