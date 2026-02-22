import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date

st.set_page_config(layout="wide")
st.title("🚀 Clean EMA Crossover Backtester")

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("Strategy Settings")

symbol = st.sidebar.text_input("Stock Symbol", "RELIANCE.NS")
start = st.sidebar.date_input("Start Date", date(2018, 1, 1))
end = st.sidebar.date_input("End Date", date.today())

short_ema = st.sidebar.slider("Short EMA", 5, 50, 20)
long_ema = st.sidebar.slider("Long EMA", 20, 200, 50)

initial_capital = st.sidebar.number_input("Initial Capital", 100000)
stop_loss_pct = st.sidebar.slider("Stop Loss %", 0.5, 10.0, 2.0) / 100

run = st.sidebar.button("Run Backtest")

# -----------------------------
# MAIN LOGIC
# -----------------------------
if run:

    data = yf.download(symbol, start=start, end=end, auto_adjust=True)

    # Fix multi-index issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data.empty:
        st.error("No data found. Check symbol or date range.")
        st.stop()

    # Calculate EMAs
    data['EMA_SHORT'] = data['Close'].ewm(span=short_ema, adjust=False).mean()
    data['EMA_LONG'] = data['Close'].ewm(span=long_ema, adjust=False).mean()

    # Generate signals
    data['Signal'] = 0
    data.loc[data['EMA_SHORT'] > data['EMA_LONG'], 'Signal'] = 1
    data['Position'] = data['Signal'].diff()

    capital = initial_capital
    position = 0
    entry_price = 0
    equity_curve = []
    trade_results = []

    for i in range(len(data)):

        price = float(data['Close'].iloc[i])

        # ENTRY
        if data['Position'].iloc[i] == 1 and position == 0:
            entry_price = price
            position = capital / entry_price
            capital = 0

        # EXIT
        elif data['Position'].iloc[i] == -1 and position > 0:
            capital = position * price
            pnl_pct = (price - entry_price) / entry_price * 100
            trade_results.append(pnl_pct)
            position = 0

        # STOP LOSS
        if position > 0 and price <= entry_price * (1 - stop_loss_pct):
            capital = position * price
            pnl_pct = (price - entry_price) / entry_price * 100
            trade_results.append(pnl_pct)
            position = 0

        equity = capital + position * price
        equity_curve.append(equity)

    data['Equity'] = equity_curve
    final_value = equity_curve[-1]

    # -----------------------------
    # PERFORMANCE METRICS
    # -----------------------------
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    days = (data.index[-1] - data.index[0]).days
    years = days / 365 if days > 0 else 1
    cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100

    wins = len([x for x in trade_results if x > 0])
    total_trades = len(trade_results)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    cumulative_max = data['Equity'].cummax()
    drawdown = (data['Equity'] - cumulative_max) / cumulative_max
    max_dd = drawdown.min() * 100

    # -----------------------------
    # DASHBOARD METRICS
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Return %", round(total_return, 2))
    col2.metric("CAGR %", round(cagr, 2))
    col3.metric("Win Rate %", round(win_rate, 2))
    col4.metric("Max Drawdown %", round(max_dd, 2))

    # -----------------------------
    # EQUITY CURVE
    # -----------------------------
    st.subheader("Equity Curve")
    st.line_chart(data['Equity'])

    # -----------------------------
    # PRICE + EMA CHART
    # -----------------------------
    st.subheader("Price Chart with EMAs")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_SHORT'],
        name=f"EMA {short_ema}"
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_LONG'],
        name=f"EMA {long_ema}"
    ))

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    st.success("Backtest completed successfully ✅")