import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="EMA Backtesting System", layout="wide")

st.title("📈 Professional EMA Crossover Backtester")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙ Strategy Configuration")

symbol = st.sidebar.text_input("Symbol (Example: RELIANCE.NS)", "RELIANCE.NS")

start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

ema_fast = st.sidebar.slider("Fast EMA", 5, 50, 20)
ema_slow = st.sidebar.slider("Slow EMA", 20, 200, 50)

initial_capital = st.sidebar.number_input("Initial Capital", value=100000.0)
position_size_percent = st.sidebar.slider("Capital Per Trade (%)", 1, 100, 100)

entry_mode = st.sidebar.selectbox(
    "Entry Execution",
    ["Same Candle Close", "Next Candle Open"]
)

use_sl = st.sidebar.toggle("Enable Stop Loss")
sl_percent = st.sidebar.number_input("Stop Loss %", value=2.0)

use_slippage = st.sidebar.toggle("Enable Slippage")
slippage_percent = st.sidebar.number_input("Slippage %", value=0.1)

run_button = st.sidebar.button("▶ Run Backtest")

if not run_button:
    st.info("Configure strategy and click Run Backtest.")
    st.stop()

# =========================
# DATA FETCH
# =========================
with st.spinner("Downloading data..."):
    data = yf.download(symbol, start=start_date, end=end_date)

if data.empty:
    st.error("No data found for this symbol/date range.")
    st.stop()

data["EMA_FAST"] = data["Close"].ewm(span=ema_fast).mean()
data["EMA_SLOW"] = data["Close"].ewm(span=ema_slow).mean()
data.dropna(inplace=True)

# =========================
# BACKTEST ENGINE
# =========================
capital = initial_capital
equity_curve = []
trades = []

position = None
quantity = 0

for i in range(1, len(data)):

    row = data.iloc[i]
    prev = data.iloc[i-1]

    # ================= ENTRY =================
    if position is None:
        if prev["EMA_FAST"] < prev["EMA_SLOW"] and row["EMA_FAST"] > row["EMA_SLOW"]:

            if entry_mode == "Same Candle Close":
                entry_price = row["Close"]
            else:
                entry_price = row["Open"]

            if use_slippage:
                entry_price *= (1 + slippage_percent / 100)

            trade_capital = capital * (position_size_percent / 100)
            quantity = trade_capital / entry_price

            stop_loss = None
            if use_sl:
                stop_loss = entry_price * (1 - sl_percent / 100)

            position = {
                "entry_price": entry_price,
                "entry_date": row.name,
                "stop_loss": stop_loss
            }

    # ================= EXIT =================
    elif position is not None:

        exit_signal = False
        exit_price = None

        # EMA CROSS EXIT
        if prev["EMA_FAST"] > prev["EMA_SLOW"] and row["EMA_FAST"] < row["EMA_SLOW"]:
            exit_signal = True
            exit_price = row["Close"]

        # STOP LOSS CHECK
        if use_sl and row["Low"] <= position["stop_loss"]:
            exit_signal = True
            exit_price = position["stop_loss"]

        if exit_signal:

            if use_slippage:
                exit_price *= (1 - slippage_percent / 100)

            pnl = (exit_price - position["entry_price"]) * quantity
            capital += pnl

            trades.append({
                "Entry Date": position["entry_date"],
                "Exit Date": row.name,
                "Entry Price": round(position["entry_price"], 2),
                "Exit Price": round(exit_price, 2),
                "PnL": round(pnl, 2)
            })

            position = None
            quantity = 0

    equity_curve.append(capital)

# =========================
# PERFORMANCE METRICS
# =========================
total_trades = len(trades)
winning_trades = len([t for t in trades if t["PnL"] > 0])
losing_trades = len([t for t in trades if t["PnL"] <= 0])

win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
total_return = (capital - initial_capital) / initial_capital * 100

years = (data.index[-1] - data.index[0]).days / 365
cagr = ((capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

# Drawdown
equity_series = pd.Series(equity_curve)
rolling_max = equity_series.cummax()
drawdown = (equity_series - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100

# =========================
# DASHBOARD
# =========================
st.subheader("📊 Performance Summary")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Final Capital", f"₹{capital:,.0f}")
col2.metric("Total Return %", f"{total_return:.2f}%")
col3.metric("Win Rate %", f"{win_rate:.2f}%")
col4.metric("CAGR %", f"{cagr:.2f}%")
col5.metric("Max Drawdown %", f"{max_drawdown:.2f}%")

# =========================
# PRICE CHART
# =========================
st.subheader("📈 Price Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
))

fig.add_trace(go.Scatter(
    x=data.index,
    y=data["EMA_FAST"],
    line=dict(width=1),
    name=f"EMA {ema_fast}"
))

fig.add_trace(go.Scatter(
    x=data.index,
    y=data["EMA_SLOW"],
    line=dict(width=1),
    name=f"EMA {ema_slow}"
))

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# =========================
# EQUITY CURVE
# =========================
st.subheader("📊 Equity Curve")

equity_df = pd.DataFrame({
    "Date": data.index[1:len(equity_curve)+1],
    "Equity": equity_curve
})

st.line_chart(equity_df.set_index("Date"))

# =========================
# TRADE LOG
# =========================
if trades:
    st.subheader("📋 Trade Log")
    st.dataframe(pd.DataFrame(trades))
else:
    st.warning("No trades generated with current settings.")