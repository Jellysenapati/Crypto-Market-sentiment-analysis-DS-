from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_sentiment_data, load_trader_data
from src.preprocessing import clean_sentiment_data, clean_trader_data, merge_sentiment_and_trades
from src.trader_analysis import risk_exposure_by_account, sentiment_vs_performance, trader_performance_summary


st.set_page_config(page_title="Crypto Sentiment Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 40%, #0b1220 100%);
            color: #e5e7eb;
        }
        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            color: #94a3b8;
            margin-bottom: 1rem;
        }
        .kpi-card {
            background: rgba(30, 41, 59, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 14px;
            padding: 14px 16px;
        }
        .kpi-label {
            color: #cbd5e1;
            font-size: 0.85rem;
            margin-bottom: 0.1rem;
        }
        .kpi-value {
            color: #f8fafc;
            font-size: 1.5rem;
            font-weight: 700;
        }
        .block-container {
            padding-top: 1.5rem;
        }
        [data-testid="stSidebar"] {
            background: #0b1220;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    processed_path = PROCESSED_DIR / "trades_enriched.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce", utc=True)
        return df

    sentiment_raw = load_sentiment_data()
    trader_raw = load_trader_data()
    sentiment = clean_sentiment_data(sentiment_raw)
    trader = clean_trader_data(trader_raw)
    return merge_sentiment_and_trades(sentiment, trader)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    sentiments = sorted(df["sentiment_classification"].dropna().unique().tolist())
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    accounts = sorted(df["account"].dropna().unique().tolist())

    selected_sentiments = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)
    selected_symbols = st.sidebar.multiselect("Symbol", symbols, default=symbols[: min(len(symbols), 10)])
    selected_accounts = st.sidebar.multiselect("Account", accounts, default=accounts)

    min_date = df["trade_date"].min()
    max_date = df["trade_date"].max()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    filtered = df[
        df["sentiment_classification"].isin(selected_sentiments)
        & df["symbol"].isin(selected_symbols if selected_symbols else symbols)
        & df["account"].isin(selected_accounts if selected_accounts else accounts)
    ]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["trade_date"].dt.date >= start_date)
            & (filtered["trade_date"].dt.date <= end_date)
        ]
    return filtered


def render_kpis(df: pd.DataFrame) -> None:
    total_trades = int(len(df))
    unique_accounts = int(df["account"].nunique())
    total_pnl = float(df["closed_pnl"].sum())
    avg_leverage = float(df["leverage"].mean()) if len(df) else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Total Trades</div>"
        f"<div class='kpi-value'>{total_trades:,}</div></div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Unique Accounts</div>"
        f"<div class='kpi-value'>{unique_accounts:,}</div></div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Total PnL</div>"
        f"<div class='kpi-value'>{total_pnl:,.2f}</div></div>",
        unsafe_allow_html=True,
    )
    col4.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Average Leverage</div>"
        f"<div class='kpi-value'>{avg_leverage:.2f}</div></div>",
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.65)",
        legend_title_text="",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.18)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.18)")
    return fig


def main() -> None:
    apply_custom_style()
    st.markdown("<div class='main-title'>Crypto Market Sentiment Analysis Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Interactive analysis of Hyperliquid trader behavior vs Fear & Greed market sentiment.</div>",
        unsafe_allow_html=True,
    )

    try:
        df = load_dashboard_data()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    if df.empty:
        st.warning("No rows available after loading.")
        st.stop()

    filtered = sidebar_filters(df)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Rows after filters: {len(filtered):,}")
    render_kpis(filtered)

    tab1, tab2, tab3 = st.tabs(["Performance", "Behavior", "Accounts"])

    with tab1:
        st.subheader("Sentiment vs Performance")
        sentiment_perf = sentiment_vs_performance(filtered)
        st.dataframe(sentiment_perf, use_container_width=True, height=280)

        fig_box = px.box(
            filtered,
            x="sentiment_classification",
            y="closed_pnl",
            color="sentiment_classification",
            title="PnL Distribution by Sentiment",
            points="outliers",
        )
        st.plotly_chart(apply_chart_theme(fig_box), use_container_width=True)

    with tab2:
        st.subheader("Leverage Behavior")
        leverage_scatter = px.scatter(
            filtered.sample(min(len(filtered), 5000), random_state=42),
            x="leverage",
            y="closed_pnl",
            color="sentiment_classification",
            hover_data=["account", "symbol", "side"],
            title="Leverage vs PnL",
            opacity=0.7,
        )
        st.plotly_chart(apply_chart_theme(leverage_scatter), use_container_width=True)

        st.subheader("Trading Activity")
        daily = (
            filtered.groupby("trade_date", as_index=False)
            .agg(trades=("trade_date", "size"), total_pnl=("closed_pnl", "sum"))
            .sort_values("trade_date")
        )
        fig_daily = px.line(daily, x="trade_date", y=["trades", "total_pnl"], title="Daily Trades and PnL")
        st.plotly_chart(apply_chart_theme(fig_daily), use_container_width=True)

    with tab3:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Top Traders")
            summary = trader_performance_summary(filtered).head(20)
            st.dataframe(summary, use_container_width=True, height=420)
        with col_right:
            st.subheader("Highest Risk Accounts")
            risk = risk_exposure_by_account(filtered).head(20)
            st.dataframe(risk, use_container_width=True, height=420)


if __name__ == "__main__":
    main()

