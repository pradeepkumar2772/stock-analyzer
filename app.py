import streamlit as st
import plotly.graph_objects as go
from datetime import date
from backtester import *
from relative_strength import relative_strength

st.set_page_config(layout="wide")
st.title("🚀 Pro EMA Backtesting Dashboard")

# Sidebar
st.sidebar.header("Strategy Controls")

symbol = st.sidebar.text_input("Stock", "RELIANCE.NS")
benchmark = st.sidebar.text_input("Benchmark", "^NSEI")

start = st.sidebar.date_input("Start", date(2018, 1, 1))
end = st.sidebar.date_input("End", date.today())

short_ema = st.sidebar.slider("Short EMA", 5, 50, 20)
long_ema = st.sidebar.slider("Long EMA", 20, 200, 50)

initial_capital = st.sidebar.number_input("Capital", 100000)

stop_loss = st.sidebar.slider("Stop Loss %", 0.5, 10.0, 2.0) / 100

run = st.sidebar.button("Run Analysis")

if run:

    data = fetch_data(symbol, start, end)
    data = add_ema(data, short_ema, long_ema)
    data = generate_signals(data)

    data, final_value, trade_results, trade_log = backtest(
        data,
        initial_capital,
        stop_loss_pct=stop_loss
    )

    metrics = performance_metrics(data, trade_results, initial_capital)

    # TOP METRICS DASHBOARD
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Return %", metrics["Total Return %"])
    col2.metric("CAGR %", metrics["CAGR %"])
    col3.metric("Win Rate %", metrics["Win Rate %"])
    col4.metric("Max DD %", metrics["Max Drawdown %"])

    # EQUITY CURVE
    st.subheader("Equity Curve")
    st.line_chart(data['Equity'])

    # CANDLESTICK
    st.subheader("Price + EMA")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    ))

    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_SHORT'], name="EMA Short"))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_LONG'], name="EMA Long"))

    st.plotly_chart(fig, use_container_width=True)

    # RELATIVE STRENGTh
    st.subheader("Relative Strength vs Benchmark")

    rs_df = relative_strength(symbol, benchmark, start, end)
    st.line_chart(rs_df)

    # TRADE LOG
    st.subheader("Trade Log")
    st.dataframe(trade_log) 