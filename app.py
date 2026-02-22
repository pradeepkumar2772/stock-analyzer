import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(layout="wide")
st.title("🚀 PK Ribbon Performance Suite")

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("Strategy Settings")

symbol = st.sidebar.text_input("Stock Symbol", "RELIANCE.NS").upper()

# Timeframe Selector
tf_options = {"1 Day": "1d", "1 Hour": "1h", "15 Min": "15m", "5 Min": "5m"}
selected_tf_label = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=0)
interval = tf_options[selected_tf_label]

# --- THE FIX: No min_value, no max_value, just raw date objects ---
start = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end = st.sidebar.date_input("End Date", date.today())

short_ema = st.sidebar.slider("Short EMA", 5, 50, 20)
long_ema = st.sidebar.slider("Long EMA", 20, 200, 50)
initial_capital = st.sidebar.number_input("Initial Capital", 100000)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Toggles & Risk")
use_sl = st.sidebar.checkbox("Enable Stop Loss", value=True)
stop_loss_pct = st.sidebar.slider("Stop Loss %", 0.5, 10.0, 2.0) / 100 if use_sl else 0
use_tp = st.sidebar.checkbox("Enable Target Profit", value=False)
tp_pct = st.sidebar.slider("Target Profit %", 1.0, 50.0, 10.0) / 100 if use_tp else 0
use_slippage = st.sidebar.checkbox("Apply Slippage", value=True)
slippage_val = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) / 100 if use_slippage else 0

run = st.sidebar.button("🚀 Run Full Analysis")

# -----------------------------
# MAIN LOGIC
# -----------------------------
if run:
    # Handle Timeframe Limits internally ONLY
    max_days = {"1d": 20000, "1h": 729, "15m": 59, "5m": 59}
    earliest_allowed = date.today() - timedelta(days=max_days[interval])
    
    # Correction logic happens here, far away from the UI widget
    final_start = start if start >= earliest_allowed else earliest_allowed
    
    if start < earliest_allowed:
        st.info(f"💡 Note: Start date adjusted to {final_start} for {selected_tf_label} data.")

    try:
        data = yf.download(symbol, start=final_start, end=end, interval=interval, auto_adjust=True)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            st.error("No data found.")
            st.stop()

        # Indicators
        data['EMA_SHORT'] = data['Close'].ewm(span=short_ema, adjust=False).mean()
        data['EMA_LONG'] = data['Close'].ewm(span=long_ema, adjust=False).mean()
        data['EMA_EXIT'] = data['Close'].ewm(span=30, adjust=False).mean()

        # Signals
        data['Signal'] = 0
        data.loc[data['EMA_SHORT'] > data['EMA_LONG'], 'Signal'] = 1
        data['Entry_Exit'] = data['Signal'].diff()

        capital = initial_capital
        shares, entry_price = 0, 0
        equity_curve, trade_results = [], []

        for i in range(len(data)):
            price = float(data['Close'].iloc[i])
            # Entry
            if data['Entry_Exit'].iloc[i] == 1 and shares == 0:
                entry_price = price * (1 + slippage_val)
                shares = capital / entry_price
                capital = 0
            # Exit
            elif shares > 0:
                sl_hit = use_sl and price <= entry_price * (1 - stop_loss_pct)
                tp_hit = use_tp and price >= entry_price * (1 + tp_pct)
                cross_exit = data['EMA_SHORT'].iloc[i] < data['EMA_EXIT'].iloc[i]

                if sl_hit or tp_hit or cross_exit:
                    exit_p = price * (1 - slippage_val)
                    capital = shares * exit_p
                    trade_results.append((exit_p - entry_price) / entry_price * 100)
                    shares = 0
            equity_curve.append(capital + (shares * price))

        data['Equity'] = equity_curve
        
        # --- Metrics Math ---
        wins = [x for x in trade_results if x > 0]
        losses = [x for x in trade_results if x <= 0]
        win_rate = (len(wins) / len(trade_results) * 100) if trade_results else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else sum(wins)
        
        days_diff = (data.index[-1] - data.index[0]).days
        years = days_diff / 365.25 if days_diff > 0 else 1
        cagr = ((equity_curve[-1] / initial_capital) ** (1 / years) - 1) * 100
        mdd = ((data['Equity'] - data['Equity'].cummax()) / data['Equity'].cummax()).min() * 100

        # --- Dashboard ---
        st.subheader("📊 Performance Scorecard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net P&L", f"₹{equity_curve[-1] - initial_capital:,.0f}")
        c2.metric("Total Return", f"{((equity_curve[-1]/initial_capital)-1)*100:.2f}%")
        c3.metric("CAGR", f"{cagr:.2f}%")
        c4.metric("Max Drawdown", f"{mdd:.2f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Success Ratio", f"{win_rate:.1f}%")
        c6.metric("Profit Factor", f"{profit_factor:.2f}")
        c7.metric("Avg Return/Trade", f"{np.mean(trade_results) if trade_results else 0:.2f}%")
        c8.metric("Total Trades", len(trade_results))

        st.line_chart(data['Equity'])
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"))
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Execution Error: {e}")