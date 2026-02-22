import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
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
    
    # RSI for filtering
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    df['long_signal'] = (df['ema20'] > df['ema50']) & (df['ema20'].shift(1) <= df['ema50'].shift(1))
    if config['use_rsi_filter']:
        mode = config['rsi_mode']
        if mode == "Greater Than": df['long_signal'] &= (df['rsi'] > config['rsi_val1'])
        elif mode == "Less Than": df['long_signal'] &= (df['rsi'] < config['rsi_val1'])
        elif mode == "Between Range": df['long_signal'] &= (df['rsi'] >= config['rsi_val1']) & (df['rsi'] <= config['rsi_val2'])
        
    df['exit_signal'] = (df['ema20'] < df['ema30']) & (df['ema20'].shift(1) >= df['ema30'].shift(1))

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            if sl_hit or tp_hit or prev['exit_signal']:
                reason = "Stop Loss" if sl_hit else ("Target Profit" if tp_hit else "EMA Cross Exit")
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
    return trades, df

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Backtesting Report Pro")
st.sidebar.title("🎗️ PK Ribbon Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
tf_limits = {"1 Day": "1d", "1 Hour": "1h", "15 Minutes": "15m", "5 Minutes": "5m"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_limits.keys()))
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2005-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

st.sidebar.divider()
st.sidebar.subheader("🛡️ Filters & Risk")
use_rsi_filter = st.sidebar.toggle("Enable RSI Filter", value=False)
rsi_mode, rsi_val1, rsi_val2 = "Greater Than", 50.0, 70.0
if use_rsi_filter:
    rsi_mode = st.sidebar.selectbox("RSI Mode", ["Greater Than", "Less Than", "Between Range"])
    rsi_val1 = st.sidebar.number_input("RSI 1", 0.0, 100.0, 50.0)
    if rsi_mode == "Between Range": rsi_val2 = st.sidebar.number_input("RSI 2", 0.0, 100.0, 70.0)

use_sl = st.sidebar.checkbox("Stop Loss", True); sl_val = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0
use_tp = st.sidebar.checkbox("Target Profit", True); tp_val = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0
slippage_val = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1)

if st.sidebar.button("🚀 Generate Report"):
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_limits[selected_tf], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            
            config = {'use_sl': use_sl, 'sl_val': sl_val, 'use_tp': use_tp, 'tp_val': tp_val, 'use_slippage': True, 'slippage_val': slippage_val, 'capital': capital, 'use_rsi_filter': use_rsi_filter, 'rsi_mode': rsi_mode, 'rsi_val1': rsi_val1, 'rsi_val2': rsi_val2}
            trades, processed_df = run_backtest(data, symbol, config)

            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                
                # Math for Reports
                wins = df_trades[df_trades['pnl_pct'] > 0]
                losses = df_trades[df_trades['pnl_pct'] <= 0]
                total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                years = (df_trades['exit_date'].iloc[-1] - pd.to_datetime(df_trades['entry_date'].iloc[0])).days / 365.25
                cagr = (((df_trades['equity'].iloc[-1] / capital) ** (1/years)) - 1) * 100 if years > 0 else 0
                peak = df_trades['equity'].cummax()
                drawdown = (df_trades['equity'] - peak) / peak
                mdd = drawdown.min() * 100
                
                # --- TAB SYSTEM ---
                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                
                with t1:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Returns (%)", f"{total_ret:.2f}%", delta_color="normal")
                    c2.metric("Max Drawdown(MDD)", f"{mdd:.2f}%", delta_color="inverse")
                    c3.metric("Total no. of Trades", len(df_trades))
                    
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Initial Capital", f"{capital:,.2f}")
                    c5.metric("Final Capital", f"{df_trades['equity'].iloc[-1]:,.2f}")
                    c6.metric("Win Ratio", f"{(len(wins)/len(df_trades)*100):.2f}%")
                    
                    c7, c8, c9 = st.columns(3)
                    c7.metric("CAGR", f"{cagr:.2f}%")
                    c8.metric("Average Return Per Trade", f"{(df_trades['pnl_pct'].mean()*100):.2f}%")
                    c9.metric("Risk-Reward Ratio", f"{(wins['pnl_pct'].mean()/abs(losses['pnl_pct'].mean())):.2f}" if not losses.empty else "N/A")

                with t2:
                    st.subheader("Performance Characteristics")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Return Metrics**")
                        st.write(f"Highest Return: {df_trades['pnl_pct'].max()*100:.2f}%")
                        st.write(f"Lowest Return: {df_trades['pnl_pct'].min()*100:.2f}%")
                        st.write(f"Avg Return (Winner): {wins['pnl_pct'].mean()*100:.2f}%")
                        st.write(f"Avg Return (Loser): {losses['pnl_pct'].mean()*100:.2f}%")
                    with col_b:
                        st.write("**Risk-Adjusted Metrics**")
                        sharpe = (df_trades['pnl_pct'].mean() / df_trades['pnl_pct'].std()) * np.sqrt(252) if len(df_trades) > 1 else 0
                        st.write(f"Sharpe Ratio: {sharpe:.2f}")
                        st.write(f"Calmar Ratio: {abs(cagr/mdd):.2f}" if mdd != 0 else "N/A")
                        st.write(f"Average Drawdown: {drawdown.mean()*100:.2f}%")

                with t3:
                    st.subheader("Equity & Drawdown Visuals")
                    fig_eq = px.line(df_trades, x='exit_date', y='equity', title="Equity Curve (Long)")
                    st.plotly_chart(fig_eq, use_container_width=True)
                    
                    fig_dd = px.area(df_trades, x='exit_date', y=drawdown*100, title="Drawdown (%)", color_discrete_sequence=['red'])
                    st.plotly_chart(fig_dd, use_container_width=True)
                    
                    st.subheader("Trades by Day of the Week")
                    df_trades['day'] = df_trades['exit_date'].dt.day_name()
                    day_stats = df_trades.groupby(['day', pnl_bool := df_trades['pnl_pct'] > 0]).size().unstack(fill_value=0)
                    st.bar_chart(day_stats)

                with t4:
                    st.subheader("Trade Audit Log")
                    st.dataframe(df_trades[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl_pct', 'exit_reason']], use_container_width=True)
                    st.download_button("Export to CSV", df_trades.to_csv().encode('utf-8'), "report.csv", "text/csv")
            else:
                st.warning("No trades found.")
    except Exception as e:
        st.error(f"Error: {e}")