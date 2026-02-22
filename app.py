import streamlit as st
import plotly.graph_objects as go
from datetime import date
from backtester import *

st.set_page_config(page_title="EMA Backtester", layout="wide")

st.title("📈 EMA Crossover Backtesting Engine")

# Sidebar Inputs
st.sidebar.header("Strategy Settings")

symbol = st.sidebar.text_input("Stock Symbol (e.g. RELIANCE.NS)", "RELIANCE.NS")
start = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end = st.sidebar.date_input("End Date", date.today())

initial_capital = st.sidebar.number_input("Initial Capital", 100000)
brokerage = st.sidebar.number_input("Brokerage %", 0.05) / 100
slippage = st.sidebar.number_input("Slippage %", 0.05) / 100
stop_loss = st.sidebar.number_input("Stop Loss %", 2.0) / 100

run = st.sidebar.button("Run Backtest")

if run:

    data = fetch_data(symbol, start, end)
    data = add_ema(data)
    data = generate_signals(data)

    data, final_value, trades, trade_log = backtest(
        data,
        initial_capital,
        brokerage,
        slippage,
        stop_loss
    )

    metrics = performance_metrics(data, initial_capital)
    suggestion = strategy_suggestion(metrics)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance Metrics")
        st.write(metrics)

    with col2:
        st.subheader("Strategy Verdict")
        st.success(suggestion)

    # Equity Curve
    st.subheader("Equity Curve")
    st.line_chart(data['Equity'])

    # Candlestick Chart
    st.subheader("Price Chart")

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_SHORT'],
        name="EMA 20"
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_LONG'],
        name="EMA 50"
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Trade Log
    st.subheader("Trade Log")
    st.dataframe(trade_log)