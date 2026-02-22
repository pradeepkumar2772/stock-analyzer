import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(layout="wide")
st.title("🚀 Professional EMA Crossover Backtester")

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("Strategy Settings")

symbol = st.sidebar.text_input("Stock Symbol", "RELIANCE.NS")

# TIMEFRAME SELECTOR
tf_options = {
    "1 Day": "1d",
    "1 Hour": "1h",
    "15 Min": "15m",
    "5 Min": "5m"
}
selected_tf_label = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=0)
interval = tf_options[selected_tf_label]

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
    # Handle Timeframe Limits for Yahoo Finance
    fetch_start = start
    if interval in ["5m", "15m"]:
        limit_date = date.today() - timedelta(days=59)
        if start < limit_date:
            fetch_start = limit_date
            st.info(f"💡 Auto-adjusted start date to {fetch_start} (Max history for {selected_tf_label})")
    elif interval == "1h":
        limit_date = date.today() - timedelta(days=729)
        if start < limit_date:
            fetch_start = limit_date
            st.info(f"💡 Auto-adjusted start date to {fetch_start} (Max history for {selected_tf_label})")

    data = yf.download(symbol, start=fetch_start, end=end, interval=interval, auto_adjust=True)

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
    data['Position_Signal'] = data['Signal'].diff()

    capital = initial_capital
    shares = 0
    entry_price = 0
    equity_curve = []
    trade_results = [] # Stores PnL % per trade

    for i in range(len(data)):
        price = float(data['Close'].iloc[i])

        # ENTRY
        if data['Position_Signal'].iloc[i] == 1 and shares == 0:
            entry_price = price
            shares = capital / entry_price
            capital = 0

        # EXIT (EMA Crossover)
        elif data['Position_Signal'].iloc[i] == -1 and shares > 0:
            capital = shares * price
            pnl_pct = (price - entry_price) / entry_price * 100
            trade_results.append(pnl_pct)
            shares = 0

        # STOP LOSS
        elif shares > 0 and price <= entry_price * (1 - stop_loss_pct):
            capital = shares * price
            pnl_pct = (price - entry_price) / entry_price * 100
            trade_results.append(pnl_pct)
            shares = 0

        current_equity = capital + (shares * price)
        equity_curve.append(current_equity)

    data['Equity'] = equity_curve
    final_value = equity_curve[-1]

    # -----------------------------
    # ADVANCED PERFORMANCE METRICS
    # -----------------------------
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    days = (data.index[-1] - data.index[0]).days
    years = days / 365.25 if days > 0 else 1
    cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100

    # Trade Statistics
    wins = [x for x in trade_results if x > 0]
    losses = [x for x in trade_results if x <= 0]
    
    win_rate = (len(wins) / len(trade_results) * 100) if trade_results else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.0001
    
    # Risk-Reward & Profit Factor
    risk_reward = avg_win / avg_loss
    profit_factor = sum(wins) / abs(sum(losses)) if losses else sum(wins)
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)

    # Drawdown
    cumulative_max = data['Equity'].cummax()
    drawdown = (data['Equity'] - cumulative_max) / cumulative_max
    max_dd = drawdown.min() * 100

    # -----------------------------
    # DASHBOARD METRICS
    # -----------------------------
    st.subheader("📊 Performance Scorecard")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Return %", f"{total_return:.2f}%")
    m2.metric("CAGR %", f"{cagr:.2f}%")
    m3.metric("Win Rate %", f"{win_rate:.1f}%")
    m4.metric("Max Drawdown %", f"{max_dd:.2f}%")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Profit Factor", f"{profit_factor:.2f}")
    m6.metric("Risk:Reward", f"1:{risk_reward:.2f}")
    m7.metric("Expectancy", f"{expectancy:.2f}%")
    m8.metric("Total Trades", len(trade_results))

    # -----------------------------
    # CHARTS
    # -----------------------------
    st.subheader("📈 Equity Growth")
    st.line_chart(data['Equity'])

    st.subheader("🕯️ Price Action & EMA Ribbon")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_SHORT'], name=f"EMA {short_ema}", line=dict(color='yellow')))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_LONG'], name=f"EMA {long_ema}", line=dict(color='red')))
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # DOWNLOAD RESULTS
    st.subheader("📝 Detailed Trade Log")
    log_df = pd.DataFrame({"PnL %": trade_results})
    st.dataframe(log_df, use_container_width=True)
    
    csv = log_df.to_csv(index=True).encode('utf-8')
    st.download_button("📥 Download Trade Log CSV", data=csv, file_name=f"{symbol}_results.csv")

    st.success("Analysis Complete ✅")