import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="EMA Backtester Pro", layout="wide")

st.title("📊 EMA Crossover Backtesting Engine")

# =====================================================
# SYMBOL LOADER (NSE + BSE)
# =====================================================

@st.cache_data(ttl=86400)
def load_nse_symbols():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    df = pd.read_csv(StringIO(response.text))
    df.columns = df.columns.str.strip()
    df = df[df["SERIES"] == "EQ"]
    return sorted([s + ".NS" for s in df["SYMBOL"].tolist()])


@st.cache_data(ttl=86400)
def load_bse_symbols():
    nse_symbols = load_nse_symbols()
    return sorted([s.replace(".NS", ".BO") for s in nse_symbols])


exchange = st.sidebar.radio("Exchange", ["NSE", "BSE"])

if exchange == "NSE":
    symbols = load_nse_symbols()
else:
    symbols = load_bse_symbols()

symbol = st.sidebar.selectbox("🔍 Search Stock", symbols)

# =====================================================
# PARAMETERS
# =====================================================

short_ema = st.sidebar.slider("Short EMA", 5, 50, 20)
long_ema = st.sidebar.slider("Long EMA", 20, 200, 50)

start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# =====================================================
# FETCH DATA
# =====================================================

@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

data = load_data(symbol, start_date, end_date)

if data.empty:
    st.error("No data found for this symbol/date range.")
    st.stop()

# =====================================================
# STRATEGY LOGIC
# =====================================================

data["EMA_Short"] = data["Close"].ewm(span=short_ema, adjust=False).mean()
data["EMA_Long"] = data["Close"].ewm(span=long_ema, adjust=False).mean()

data["Signal"] = 0
data["Signal"] = np.where(
    data["EMA_Short"] > data["EMA_Long"], 1, 0
)

data["Position"] = data["Signal"].diff()

# Strategy Returns
data["Market Return"] = data["Close"].pct_change()
data["Strategy Return"] = data["Market Return"] * data["Signal"].shift(1)

data["Equity Curve"] = (1 + data["Strategy Return"]).cumprod()

# =====================================================
# METRICS
# =====================================================

total_return = data["Equity Curve"].iloc[-1] - 1
years = (end_date - start_date).days / 365
cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

# Win Rate
trades = data[data["Position"] == 1].index
wins = 0
losses = 0

for i in range(len(trades) - 1):
    entry = data.loc[trades[i], "Close"]
    exit = data.loc[trades[i + 1], "Close"]
    if exit > entry:
        wins += 1
    else:
        losses += 1

win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

# Max Drawdown
roll_max = data["Equity Curve"].cummax()
drawdown = data["Equity Curve"] / roll_max - 1
max_dd = drawdown.min()

# =====================================================
# DASHBOARD METRICS
# =====================================================

col1, col2, col3 = st.columns(3)

col1.metric("Total Return %", f"{total_return*100:.2f}%")
col2.metric("CAGR %", f"{cagr*100:.2f}%")
col3.metric("Win Rate %", f"{win_rate:.2f}%")

st.metric("Max Drawdown %", f"{max_dd*100:.2f}%")

# =====================================================
# PRICE + EMA CHART
# =====================================================

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
    y=data["EMA_Short"],
    line=dict(width=1),
    name=f"EMA {short_ema}"
))

fig.add_trace(go.Scatter(
    x=data.index,
    y=data["EMA_Long"],
    line=dict(width=1),
    name=f"EMA {long_ema}"
))

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# EQUITY CURVE
# =====================================================

st.subheader("📈 Strategy Equity Curve")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=data.index,
    y=data["Equity Curve"],
    name="Equity Curve"
))

st.plotly_chart(fig2, use_container_width=True)