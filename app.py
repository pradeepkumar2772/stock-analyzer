import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from datetime import datetime, date, timedelta

# --- 1. CORE DATA STRUCTURES ---
@dataclass
class Trade:
    symbol: str
    direction: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime = None
    exit_price: float = None
    exit_reason: str = None
    pnl_pct: float = 0.0

# --- 2. BACKTEST ENGINE ---
def run_backtest(df, symbol, config):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    df['long_signal'] = (df['ema20'] > df['ema50']) & (df['ema20'].shift(1) <= df['ema50'].shift(1))
    df['exit_signal'] = (df['ema20'] < df['ema30']) & (df['ema20'].shift(1) >= df['ema30'].shift(1))

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        if active_trade:
            sl_hit, tp_hit = False, False
            if config['use_sl']:
                sl_p = active_trade.entry_price * (1 - config['sl_val'] / 100)
                sl_hit = current['low'] <= sl_p
            if config['use_tp']:
                tp_p = active_trade.entry_price * (1 + config['tp_val'] / 100)
                tp_hit = current['high'] >= tp_p
            indicator_exit = prev['exit_signal']

            if sl_hit or tp_hit or indicator_exit:
                reason = "Stop Loss" if sl_hit else ("Target" if tp_hit else "EMA Cross Exit")
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None
        elif prev['long_signal']:
            entry_p = current['open'] * (1 + slippage)
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=entry_p)
    return trades, df

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="PK Ribbon Performance Pro")

st.sidebar.title("🎗️ PK Ribbon Engine")
symbol = st.sidebar.text_input("Symbol", value="RELIANCE.NS").upper()

tf_limits = {
    "1 Minute": {"val": "1m", "max_days": 7},
    "5 Minutes": {"val": "5m", "max_days": 59},
    "15 Minutes": {"val": "15m", "max_days": 59},
    "1 Hour": {"val": "1h", "max_days": 729},
    "1 Day": {"val": "1d", "max_days": 20000},
}

selected_tf_label = st.sidebar.selectbox("Select Timeframe", list(tf_limits.keys()), index=4)
selected_tf = tf_limits[selected_tf_label]["val"]
max_days_allowed = tf_limits[selected_tf_label]["max_days"]

capital = st.sidebar.number_input("Initial Capital", value=100000)
user_start = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
user_end = st.sidebar.date_input("End Date", value=date.today())

st.sidebar.divider()
st.sidebar.subheader("⚙️ Toggles")
use_sl = st.sidebar.checkbox("Enable Stop Loss", value=True)
sl_val = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0
use_tp = st.sidebar.checkbox("Enable Target Profit", value=True)
tp_val = st.sidebar.slider("Target %", 1.0, 100.0, 25.0) if use_tp else 0
use_slippage = st.sidebar.checkbox("Apply Slippage", value=True)
slippage_val = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slippage else 0

if st.sidebar.button("🚀 Run Backtest"):
    earliest_allowed = date.today() - timedelta(days=max_days_allowed)
    final_start = user_start if user_start >= earliest_allowed else earliest_allowed
    
    try:
        with st.spinner('Calculating Statistics...'):
            data = yf.download(symbol, start=final_start, end=user_end, interval=selected_tf, auto_adjust=True)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data.columns = [str(col).lower() for col in data.columns]
                data = data.dropna()
                
                config = {'use_sl': use_sl, 'sl_val': sl_val, 'use_tp': use_tp, 'tp_val': tp_val, 'use_slippage': use_slippage, 'slippage_val': slippage_val, 'capital': capital}
                trades, processed_df = run_backtest(data, symbol, config)

                if trades:
                    df_trades = pd.DataFrame([vars(t) for t in trades])
                    
                    # --- PERFORMANCE MATH ---
                    wins = df_trades[df_trades['pnl_pct'] > 0]
                    losses = df_trades[df_trades['pnl_pct'] <= 0]
                    win_rate = (len(wins) / len(df_trades)) * 100
                    
                    avg_win = wins['pnl_pct'].mean() if not wins.empty else 0
                    avg_loss = abs(losses['pnl_pct'].mean()) if not losses.empty else 0.0001
                    risk_reward = avg_win / avg_loss
                    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
                    
                    total_profit = wins['pnl_pct'].sum()
                    total_loss = abs(losses['pnl_pct'].sum())
                    profit_factor = total_profit / total_loss if total_loss != 0 else total_profit
                    
                    final_equity = capital * (1 + df_trades['pnl_pct']).cumprod().iloc[-1]
                    years = (processed_df.index[-1] - processed_df.index[0]).days / 365.25
                    cagr = ((final_equity / capital) ** (1 / (years if years > 0 else 1)) - 1) * 100
                    
                    df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                    peak = df_trades['equity'].cummax()
                    mdd = ((df_trades['equity'] - peak) / peak).min() * 100

                    # --- UI DISPLAY ---
                    st.subheader("📊 Strategy Performance Scorecard")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("CAGR", f"{cagr:.2f}%")
                    c2.metric("Success Ratio", f"{win_rate:.1f}%")
                    c3.metric("Max Drawdown", f"{mdd:.2f}%")
                    c4.metric("Profit Factor", f"{profit_factor:.2f}")

                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("Risk:Reward", f"1:{risk_reward:.2f}")
                    c6.metric("Expectancy", f"{expectancy*100:.2f}%")
                    c7.metric("Avg Return/Trade", f"{df_trades['pnl_pct'].mean()*100:.2f}%")
                    c8.metric("Total Trades", len(df_trades))

                    # Chart
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=processed_df.index, open=processed_df['open'], high=processed_df['high'], low=processed_df['low'], close=processed_df['close'], name="Price"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['ema20'], name="EMA 20", line=dict(color='yellow')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['ema50'], name="EMA 50", line=dict(color='red')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_trades['exit_date'], y=df_trades['equity'], name="Equity Curve", fill='tozeroy', line=dict(color='#00ffcc')), row=2, col=1)
                    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(df_trades, use_container_width=True)
                    
                    csv = df_trades.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download CSV", data=csv, file_name=f"{symbol}_report.csv", mime='text/csv')
                else:
                    st.warning("No trades found.")
            else:
                st.error("No data found.")
    except Exception as e:
        st.error(f"Error: {e}")