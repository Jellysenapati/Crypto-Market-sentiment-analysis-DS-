from __future__ import annotations

import logging

from src.data_loader import load_sentiment_data, load_trader_data
from src.insights import generate_insights_text, write_insights_report
from src.preprocessing import clean_sentiment_data, clean_trader_data, merge_sentiment_and_trades
from src.sentiment_analysis import (
    extreme_sentiment_occurrences,
    fear_vs_greed_frequency,
    sentiment_distribution,
    sentiment_timeline,
)
from src.trader_analysis import (
    market_behavior_summary,
    risk_exposure_by_account,
    sentiment_vs_performance,
    trader_clustering,
    trader_performance_summary,
)
from src.utils import (
    CHARTS_DIR,
    INSIGHTS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    ensure_directories,
    save_dataframe,
    save_json,
    setup_logging,
)
from src.visualization import (
    plot_buy_sell_pie,
    plot_correlation_heatmap,
    plot_leverage_vs_pnl,
    plot_pnl_distribution,
    plot_sentiment_boxplot,
    plot_sentiment_distribution,
    plot_sentiment_timeline,
    plotly_interactive_sentiment_pnl,
)

LOGGER = logging.getLogger(__name__)


def run_pipeline() -> None:
    setup_logging()
    ensure_directories([PROCESSED_DATA_DIR, CHARTS_DIR, REPORTS_DIR, INSIGHTS_DIR])

    sentiment_raw = load_sentiment_data()
    trader_raw = load_trader_data()

    sentiment = clean_sentiment_data(sentiment_raw)
    trader = clean_trader_data(trader_raw)
    trades = merge_sentiment_and_trades(sentiment, trader)

    sentiment_dist = sentiment_distribution(sentiment)
    sentiment_bucket = fear_vs_greed_frequency(sentiment)
    sentiment_trend = sentiment_timeline(sentiment)
    extreme_events = extreme_sentiment_occurrences(sentiment)

    trader_summary = trader_performance_summary(trades)
    behavior = market_behavior_summary(trades)
    sentiment_perf = sentiment_vs_performance(trades)
    risk_summary = risk_exposure_by_account(trades)
    clustered_traders = trader_clustering(trader_summary)

    save_dataframe(sentiment, PROCESSED_DATA_DIR / "sentiment_cleaned.csv")
    save_dataframe(trader, PROCESSED_DATA_DIR / "trader_cleaned.csv")
    save_dataframe(trades, PROCESSED_DATA_DIR / "trades_enriched.csv")
    save_dataframe(sentiment_dist, PROCESSED_DATA_DIR / "sentiment_distribution.csv")
    save_dataframe(sentiment_bucket, PROCESSED_DATA_DIR / "fear_vs_greed_frequency.csv")
    save_dataframe(sentiment_perf, PROCESSED_DATA_DIR / "sentiment_performance_summary.csv")
    save_dataframe(extreme_events, PROCESSED_DATA_DIR / "extreme_sentiment_events.csv")
    save_dataframe(trader_summary, PROCESSED_DATA_DIR / "trader_performance_summary.csv")
    save_dataframe(risk_summary, PROCESSED_DATA_DIR / "account_risk_exposure.csv")
    save_dataframe(clustered_traders, PROCESSED_DATA_DIR / "trader_clusters.csv")
    save_dataframe(behavior["side_performance"], PROCESSED_DATA_DIR / "side_performance.csv")
    save_dataframe(behavior["symbol_activity"], PROCESSED_DATA_DIR / "symbol_activity.csv")
    save_dataframe(behavior["daily_activity"], PROCESSED_DATA_DIR / "daily_activity.csv")

    plot_sentiment_distribution(sentiment_dist, CHARTS_DIR)
    plot_sentiment_timeline(sentiment_trend, CHARTS_DIR)
    plot_pnl_distribution(trades, CHARTS_DIR)
    plot_leverage_vs_pnl(trades, CHARTS_DIR)
    plot_sentiment_boxplot(trades, CHARTS_DIR)
    plot_correlation_heatmap(trades, CHARTS_DIR)
    plot_buy_sell_pie(behavior["side_performance"], CHARTS_DIR)
    plotly_interactive_sentiment_pnl(trades, CHARTS_DIR)

    report_text = generate_insights_text(sentiment_perf, trader_summary, risk_summary)
    write_insights_report(report_text, REPORTS_DIR / "insights_report.md")

    quick_insights = {
        "total_trades": int(len(trades)),
        "unique_accounts": int(trades["account"].nunique()),
        "total_pnl": float(trades["closed_pnl"].sum()),
        "avg_leverage": float(trades["leverage"].mean()),
        "best_sentiment_by_avg_pnl": sentiment_perf.sort_values("avg_pnl", ascending=False)
        .head(1)["sentiment_classification"]
        .squeeze(),
    }
    save_json(quick_insights, INSIGHTS_DIR / "summary_metrics.json")
    LOGGER.info("Pipeline complete. Outputs saved in outputs/ and data/processed/")


if __name__ == "__main__":
    run_pipeline()

