# app.py
# -------------------------------------------------------
# Multi-stock Streamlit analysis dashboard.
# Run with:  streamlit run app.py
# -------------------------------------------------------
 
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from scipy import stats
from datetime import date, timedelta
import math
 
# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
 
st.markdown("""
    <style>
    [data-testid="metric-container"] {
        background-color: #1A1F2E;
        border: 1px solid #2E3450;
        border-radius: 8px;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)
 
st.title("Stock Analysis Dashboard")
 
# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header(" Settings")
 
raw_input = st.sidebar.text_input(
    "Stock Tickers (2–5, comma-separated)",
    value="NUE, STLD, ZEUS"
)
 
default_start = date.today() - timedelta(days=365 * 2)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(2000, 1, 1))
end_date   = st.sidebar.date_input("End Date",   value=date.today(),  min_value=date(2000, 1, 2))
 
ma_window  = st.sidebar.slider("Moving Average Window (days)", 5, 200, 50, 5)
vol_window = st.sidebar.slider("Rolling Volatility Window (days)", 10, 120, 30, 5)
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=4.5, step=0.1
) / 100
 
# ── About / Methodology expander ─────────────────────────
with st.sidebar.expander(" About & Methodology"):
    st.markdown("""
**What this app does:**  
Download historical adjusted closing prices for 2–5 stocks plus the S&P 500 benchmark, then provide price, return, risk, distribution, and correlation analysis.
 
**Key assumptions:**
- Returns are simple arithmetic daily returns: `(Pₜ - Pₜ₋₁) / Pₜ₋₁`
- Annualization uses **252 trading days**
- Annualized volatility = daily std × √252
- Equal-weight portfolio = average of selected stocks' daily returns
 
**Data source:** Yahoo Finance via `yfinance`  
**Benchmark:** S&P 500 (`^GSPC`)
""")
 
# ── Input validation ──────────────────────────────────────
tickers_raw = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
tickers_raw = list(dict.fromkeys(tickers_raw))  # deduplicate, preserve order
 
if len(tickers_raw) < 2:
    st.error("Please enter **at least 2** stock tickers.")
    st.stop()
if len(tickers_raw) > 5:
    st.error("Please enter **no more than 5** stock tickers.")
    st.stop()
 
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()
 
if (end_date - start_date).days < 365:
    st.error("Date range must be **at least 1 year**.")
    st.stop()
 
# ── Data download ─────────────────────────────────────────
@st.cache_data(show_spinner="Downloading market data…", ttl=3600)
def load_prices(tickers: list[str], start: date, end: date) -> tuple[pd.DataFrame, list[str]]:
    """
    Download adjusted close prices for tickers + S&P 500.
    Returns (prices_df, failed_tickers).
    """
    all_tickers = tickers + ["^GSPC"]
    raw = yf.download(all_tickers, start=start, end=end, progress=False, auto_adjust=True)
 
    if raw.empty: # type: ignore
        return pd.DataFrame(), all_tickers
 
    # Extract 'Close' level
    if isinstance(raw.columns, pd.MultiIndex): # type: ignore
        prices = raw["Close"].copy() # type: ignore
    else:
        prices = raw[["Close"]].copy() # type: ignore
        prices.columns = [tickers[0]]
 
    prices.index = pd.to_datetime(prices.index)
 
    # Identify failed / nearly empty tickers
    failed = []
    for t in all_tickers:
        if t not in prices.columns:
            failed.append(t)
        elif prices[t].isna().mean() > 0.05:
            failed.append(t)
 
    return prices, failed # type: ignore
 
 
with st.spinner("Fetching data…"):
    try:
        prices, failed = load_prices(tickers_raw, start_date, end_date)
    except Exception as e:
        st.error(f"Download error: {e}")
        st.stop()
 
if prices.empty:
    st.error("No data returned. Check your tickers and try again.")
    st.stop()
 
# Warn / drop failed tickers
user_failed = [t for t in failed if t != "^GSPC"]
if user_failed:
    st.warning(
        f"The following ticker(s) had insufficient data and were dropped: "
        f"**{', '.join(user_failed)}**"
    )
 
valid_tickers = [t for t in tickers_raw if t not in failed and t in prices.columns]
 
if len(valid_tickers) < 2:
    st.error("Fewer than 2 valid tickers remain after dropping bad data. Please try different tickers.")
    st.stop()
 
# Truncate to overlapping date range across valid tickers + benchmark
keep_cols = valid_tickers + (["^GSPC"] if "^GSPC" in prices.columns else [])
prices = prices[keep_cols].dropna(how="all")
 
overlap_start = prices[valid_tickers].dropna(how="any").index.min()
overlap_end   = prices[valid_tickers].dropna(how="any").index.max()
 
if overlap_start > prices.index.min() or overlap_end < prices.index.max():
    st.info(
        f"Data truncated to overlapping range: "
        f"**{overlap_start.date()}** → **{overlap_end.date()}**"
    )
    prices = prices.loc[overlap_start:overlap_end]
 
prices = prices.dropna(subset=valid_tickers)
 
# ── Compute returns ───────────────────────────────────────
returns = prices[valid_tickers].pct_change().dropna()
sp500_returns = prices["^GSPC"].pct_change().dropna() if "^GSPC" in prices.columns else None
 
# Equal-weight portfolio daily return
returns["_EWP"] = returns[valid_tickers].mean(axis=1)
 
# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    " Price & Returns",
    " Risk & Distribution",
    " Correlation & Diversification",
    " Summary Statistics",
])
 
# ═══════════════════════════════════════════════════════════
# TAB 1 — Price & Returns
# ═══════════════════════════════════════════════════════════
with tab1:
    st.header("Price & Return Analysis")
 
    # ── Price chart with multiselect ─────────────────────
    st.subheader("Adjusted Closing Prices")
    selected_for_price = st.multiselect(
        "Select stocks to display",
        options=valid_tickers,
        default=valid_tickers,
        key="price_select"
    )
    if selected_for_price:
        fig_price = go.Figure()
        for t in selected_for_price:
            fig_price.add_trace(go.Scatter(
                x=prices.index, y=prices[t], mode="lines", name=t
            ))
        fig_price.update_layout(
            title="Adjusted Closing Prices",
            xaxis_title="Date", yaxis_title="Price (USD)",
            template="plotly_dark", height=450, hovermode="x unified"
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Select at least one stock to display the price chart.")
 
    # ── Cumulative wealth index ($10,000) ─────────────────
    st.subheader("Cumulative Wealth Index (Starting $10,000)")
 
    wealth = (1 + returns[valid_tickers]).cumprod() * 10_000
    wealth["Equal-Weight Portfolio"] = (1 + returns["_EWP"]).cumprod() * 10_000
    if sp500_returns is not None:
        sp500_aligned = sp500_returns.reindex(returns.index).fillna(0)
        wealth["S&P 500"] = (1 + sp500_aligned).cumprod() * 10_000
 
    fig_wealth = go.Figure()
    for col in wealth.columns:
        dash = "dash" if col in ["S&P 500", "Equal-Weight Portfolio"] else "solid"
        fig_wealth.add_trace(go.Scatter(
            x=wealth.index, y=wealth[col], mode="lines", name=col,
            line=dict(dash=dash)
        ))
    fig_wealth.update_layout(
        title="Growth of $10,000 Investment",
        xaxis_title="Date", yaxis_title="Portfolio Value (USD)",
        template="plotly_dark", height=450, hovermode="x unified",
        yaxis_tickprefix="$"
    )
    st.plotly_chart(fig_wealth, use_container_width=True)
 
    # ── Rolling volatility ────────────────────────────────
    st.subheader(f"Rolling {vol_window}-Day Annualized Volatility")
    roll_vol = returns[valid_tickers].rolling(vol_window).std() * math.sqrt(252)
 
    fig_rvol = go.Figure()
    for t in valid_tickers:
        fig_rvol.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol[t], mode="lines", name=t
        ))
    fig_rvol.update_layout(
        title=f"Rolling {vol_window}-Day Annualized Volatility",
        xaxis_title="Date", yaxis_title="Annualized Volatility",
        yaxis_tickformat=".0%", template="plotly_dark", height=400,
        hovermode="x unified"
    )
    st.plotly_chart(fig_rvol, use_container_width=True)
 
 
# ═══════════════════════════════════════════════════════════
# TAB 2 — Risk & Distribution
# ═══════════════════════════════════════════════════════════
with tab2:
    st.header("Risk & Distribution Analysis")
 
    dist_ticker = st.selectbox("Select stock for distribution analysis", valid_tickers, key="dist_stock")
    r = returns[dist_ticker].dropna()
 
    # ── Histogram / Q-Q toggle ────────────────────────────
    plot_type = st.radio("Plot type", ["Histogram + Normal Fit", "Q-Q Plot"], horizontal=True)
 
    if plot_type == "Histogram + Normal Fit":
        mu, sigma = stats.norm.fit(r)
        x_range = np.linspace(float(r.min()), float(r.max()), 300)
 
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=r, nbinsx=60, histnorm="probability density",
            marker_color="mediumpurple", opacity=0.75, name="Daily Returns"
        ))
        fig_dist.add_trace(go.Scatter(
            x=x_range, y=stats.norm.pdf(x_range, mu, sigma),
            mode="lines", name="Normal Fit",
            line=dict(color="red", width=2)
        ))
        fig_dist.update_layout(
            title=f"{dist_ticker} — Daily Return Distribution",
            xaxis_title="Daily Return", yaxis_title="Density",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True)
 
    else:  # Q-Q Plot
        (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers", name="Quantiles",
            marker=dict(color="mediumpurple", size=4)
        ))
        fig_qq.add_trace(go.Scatter(
            x=[min(osm), max(osm)],
            y=[slope * min(osm) + intercept, slope * max(osm) + intercept],
            mode="lines", name="Normal Line",
            line=dict(color="red", width=2)
        ))
        fig_qq.update_layout(
            title=f"{dist_ticker} — Q-Q Plot vs Normal Distribution",
            xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig_qq, use_container_width=True)
 
    # ── Jarque-Bera test ──────────────────────────────────
    jb_stat, jb_p = stats.jarque_bera(r)
    verdict = " Fails to reject normality (p ≥ 0.05)" if jb_p >= 0.05 else " Rejects normality (p < 0.05)" # type: ignore
    st.info(f"**Jarque-Bera Test** — Statistic: {jb_stat:.2f} | p-value: {jb_p:.4f} | {verdict}")
 
    # ── Boxplot ───────────────────────────────────────────
    st.subheader("Daily Return Distribution — Boxplot Comparison")
    fig_box = go.Figure()
    for t in valid_tickers:
        fig_box.add_trace(go.Box(
            y=returns[t], name=t, boxpoints="outliers"
        ))
    fig_box.update_layout(
        title="Boxplot of Daily Returns",
        yaxis_title="Daily Return", yaxis_tickformat=".1%",
        template="plotly_dark", height=400
    )
    st.plotly_chart(fig_box, use_container_width=True)
 
 
# ═══════════════════════════════════════════════════════════
# TAB 3 — Correlation & Diversification
# ═══════════════════════════════════════════════════════════
with tab3:
    st.header("Correlation & Diversification")
 
    # ── Correlation heatmap ───────────────────────────────
    st.subheader("Correlation Heatmap")
    corr = returns[valid_tickers].corr()
 
    fig_heat = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Correlation")
    ))
    fig_heat.update_layout(
        title="Pairwise Return Correlation",
        template="plotly_dark", height=420
    )
    st.plotly_chart(fig_heat, use_container_width=True)
 
    st.divider()
 
    # ── Scatter plot ──────────────────────────────────────
    st.subheader("Return Scatter Plot")
    col_a, col_b = st.columns(2)
    with col_a:
        scatter_x = st.selectbox("Stock A (x-axis)", valid_tickers, index=0, key="sc_x")
    with col_b:
        remaining = [t for t in valid_tickers if t != scatter_x]
        scatter_y = st.selectbox("Stock B (y-axis)", remaining, index=0, key="sc_y")
 
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=returns[scatter_x], y=returns[scatter_y],
        mode="markers", name=f"{scatter_x} vs {scatter_y}",
        marker=dict(size=4, opacity=0.5, color="mediumpurple")
    ))
    # Trend line
    m, b, *_ = stats.linregress(returns[scatter_x], returns[scatter_y])
    x_fit = np.linspace(returns[scatter_x].min(), returns[scatter_x].max(), 100)
    fig_scatter.add_trace(go.Scatter(
        x=x_fit, y=m * x_fit + b, mode="lines",
        name="OLS Trend", line=dict(color="red", width=2, dash="dash")
    ))
    fig_scatter.update_layout(
        title=f"Daily Returns: {scatter_x} vs {scatter_y}",
        xaxis_title=f"{scatter_x} Daily Return",
        yaxis_title=f"{scatter_y} Daily Return",
        xaxis_tickformat=".1%", yaxis_tickformat=".1%",
        template="plotly_dark", height=420
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
 
    st.divider()
 
    # ── Rolling correlation ───────────────────────────────
    st.subheader("Rolling Correlation")
    col_c, col_d, col_e = st.columns(3)
    with col_c:
        rc_a = st.selectbox("Stock A", valid_tickers, index=0, key="rc_a")
    with col_d:
        rc_remaining = [t for t in valid_tickers if t != rc_a]
        rc_b = st.selectbox("Stock B", rc_remaining, index=0, key="rc_b")
    with col_e:
        rc_window = st.selectbox("Rolling Window (days)", [30, 60, 90, 120], index=1, key="rc_win")
 
    roll_corr = returns[rc_a].rolling(rc_window).corr(returns[rc_b])
    fig_rc = go.Figure()
    fig_rc.add_trace(go.Scatter(
        x=roll_corr.index, y=roll_corr, mode="lines",
        name=f"{rc_window}-day rolling corr", line=dict(color="gold")
    ))
    fig_rc.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_rc.update_layout(
        title=f"Rolling {rc_window}-Day Correlation: {rc_a} vs {rc_b}",
        xaxis_title="Date", yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1]),
        template="plotly_dark", height=380
    )
    st.plotly_chart(fig_rc, use_container_width=True)
 
    st.divider()
 
    # ── Two-asset portfolio explorer ──────────────────────
    st.subheader("Two-Asset Portfolio Explorer")
    st.markdown("""
> **What this shows:** When you combine two stocks into a portfolio, the resulting volatility
> is almost always *lower* than a simple weighted average of their individual volatilities —
> provided their correlation is less than 1. The curve below illustrates this diversification
> benefit. The dip in the middle means you can reduce risk without sacrificing proportional return.
> **The lower the correlation between the two stocks, the deeper the dip.**
""")
 
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        port_a = st.selectbox("Stock A", valid_tickers, index=0, key="port_a")
    with col_p2:
        port_b_opts = [t for t in valid_tickers if t != port_a]
        port_b = st.selectbox("Stock B", port_b_opts, index=0, key="port_b")
 
    w_a = st.slider(f"Weight on {port_a} (%)", 0, 100, 50, 1) / 100
    w_b = 1 - w_a
 
    ra = returns[port_a]
    rb = returns[port_b]
 
    ann_ret_a  = ra.mean() * 252
    ann_ret_b  = rb.mean() * 252
    ann_vol_a  = ra.std() * math.sqrt(252)
    ann_vol_b  = rb.std() * math.sqrt(252)
    corr_ab    = ra.corr(rb)
 
    def port_stats(wa):
        wb = 1 - wa
        ret = wa * ann_ret_a + wb * ann_ret_b
        vol = math.sqrt(
            wa**2 * ann_vol_a**2 +
            wb**2 * ann_vol_b**2 +
            2 * wa * wb * ann_vol_a * ann_vol_b * corr_ab
        )
        return ret, vol
 
    cur_ret, cur_vol = port_stats(w_a)
 
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Portfolio Return ({w_a:.0%} / {w_b:.0%})", f"{cur_ret:.2%}")
    m2.metric("Portfolio Volatility", f"{cur_vol:.2%}")
    m3.metric(f"Correlation ({port_a} & {port_b})", f"{corr_ab:.3f}")
 
    weights_range = np.linspace(0, 1, 201)
    vols_curve = [port_stats(w)[1] for w in weights_range]
    rets_curve = [port_stats(w)[0] for w in weights_range]
 
    fig_port = go.Figure()
    fig_port.add_trace(go.Scatter(
        x=weights_range * 100, y=vols_curve,
        mode="lines", name="Volatility Curve",
        line=dict(color="cyan", width=2)
    ))
    # Mark current slider position
    fig_port.add_trace(go.Scatter(
        x=[w_a * 100], y=[cur_vol],
        mode="markers", name="Current Weight",
        marker=dict(color="red", size=12, symbol="star")
    ))
    # Mark individual stocks
    fig_port.add_trace(go.Scatter(
        x=[0, 100], y=[ann_vol_b, ann_vol_a],
        mode="markers+text", name="Individual Stocks",
        marker=dict(color="yellow", size=10),
        text=[port_b, port_a], textposition=["top center", "top center"]
    ))
    fig_port.update_layout(
        title=f"Two-Asset Portfolio Volatility: {port_a} vs {port_b}",
        xaxis_title=f"Weight on {port_a} (%)",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".1%",
        template="plotly_dark", height=420
    )
    st.plotly_chart(fig_port, use_container_width=True)
 
 
# ═══════════════════════════════════════════════════════════
# TAB 4 — Summary Statistics
# ═══════════════════════════════════════════════════════════
with tab4:
    st.header("Summary Statistics")
 
    def compute_stats(ret_series: pd.Series, label: str) -> dict:
        clean = ret_series.dropna()
        return {
            "Ticker": label,
            "Ann. Return": f"{clean.mean() * 252:.2%}",
            "Ann. Volatility": f"{clean.std() * math.sqrt(252):.2%}",
            "Skewness": f"{float(clean.skew()):.3f}", # type: ignore
            "Kurtosis (excess)": f"{float(clean.kurtosis()):.3f}", # type: ignore
            "Min Daily Return": f"{float(clean.min()):.3%}",
            "Max Daily Return": f"{float(clean.max()):.3%}",
        }
 
    rows = [compute_stats(returns[t], t) for t in valid_tickers]
    if sp500_returns is not None:
        sp_aligned = sp500_returns.reindex(returns.index).dropna()
        rows.append(compute_stats(sp_aligned, "S&P 500 (^GSPC)"))
 
    stats_df = pd.DataFrame(rows).set_index("Ticker")
    st.dataframe(stats_df, use_container_width=True)
 
    st.caption(
        "Annualization uses 252 trading days. Returns are simple arithmetic daily returns. "
        "Kurtosis shown is excess kurtosis (normal distribution = 0)."
    )
 
    st.divider()
 
    # ── Per-ticker key metrics ────────────────────────────
    st.subheader("Individual Stock Metrics")
    for t in valid_tickers:
        r_t = returns[t].dropna()
        ann_r = r_t.mean() * 252
        ann_v = r_t.std() * math.sqrt(252)
        sharpe = (ann_r - risk_free_rate) / ann_v
        with st.expander(f"{t} — Sharpe: {sharpe:.2f}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ann. Return",     f"{ann_r:.2%}")
            c2.metric("Ann. Volatility", f"{ann_v:.2%}")
            c3.metric("Sharpe Ratio",    f"{sharpe:.2f}")
            c4.metric("Skewness",        f"{float(r_t.skew()):.3f}") # type: ignore
 
    # ── CSV download ──────────────────────────────────────
    st.divider()
    st.subheader("Download Data")
    csv = prices[valid_tickers].to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download Prices as CSV",
        data=csv,
        file_name="stock_prices.csv",
        mime="text/csv"
    )
 
    with st.expander("View Raw Price Data (last 60 rows)"):
        st.dataframe(prices[valid_tickers].tail(60), use_container_width=True)
