import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from datetime import datetime

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
    slippage = config['slippage'] / 100
    
    # --- PK EMA RIBBON LOGIC ---
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Entry Signal: 20 crosses above 50
    # Exit Signal: 20 crosses below 30
    df['long_signal'] = (df['ema20'] > df['ema50']) & (df['ema20'].shift(1) <= df['ema50'].shift(1))
    df['exit_signal'] = (df['ema20'] < df['ema30']) & (df['ema20'].shift(1) >= df['ema30'].shift(1))

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # --- EXIT LOGIC ---
        if active_trade:
            # Check for Price-based exits (SL/TP) or Indicator-based exit
            sl_price = active_trade.entry_price * (1 - config['sl'] / 100)
            tp_price = active_trade.entry_price * (1 + config['tp'] / 100)
            
            sl_hit = current['low'] <= sl_price
            tp_hit = current['high'] >= tp_price
            indicator_exit = prev['exit_signal'] # EMA 20 cross below 30

            if sl_hit or tp_hit or indicator_exit:
                reason = "Stop Loss" if sl_hit else ("Target" if tp_hit else "EMA Cross Exit")
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None

        # --- ENTRY LOGIC ---
        elif prev['long_signal']:
            entry_p = current['open'] * (1 + slippage)
            active_trade = Trade(
                symbol=symbol,
                direction="Long",
                entry_date=current.name,
                entry_price=entry_p
            )
            
    return trades, df

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="PK EMA Ribbon Strategy")

st.sidebar.title("🎗️ PK EMA Ribbon")
symbol = st.sidebar.text_input("Symbol (Yahoo Finance)", value="RELIANCE.NS")
capital = st.sidebar.number_input("Initial Capital", value=100000)
st.sidebar.divider()
st.sidebar.info("Entry: EMA 20 ⬆️ EMA 50\n\nExit: EMA 20 ⬇️ EMA 30")
sl_pct = st.sidebar.slider("Safety Stop Loss %", 0.5, 15.0, 5.0)
tp_pct = st.sidebar.slider("Safety Target %", 1.0, 50.0, 15.0)
slippage_pct = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1)

if st.sidebar.button("🚀 Run Strategy"):
    try:
        data = yf.download(symbol, period="3y", interval="1d")
        if data.empty:
            st.error("Invalid Symbol")
        else:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            data = data.dropna()
            
            config = {'sl': sl_pct, 'tp': tp_pct, 'slippage': slippage_pct, 'capital': capital}
            trades, processed_df = run_backtest(data, symbol, config)

            if not trades:
                st.warning("No trades found for this period.")
            else:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                
                # Metrics
                win_rate = (len(df_trades[df_trades['pnl_pct'] > 0]) / len(df_trades)) * 100
                total_ret = (df_trades['pnl_pct'] + 1).prod() - 1
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Return", f"{total_ret*100:.1f}%")
                c2.metric("Win Rate", f"{win_rate:.1f}%")
                c3.metric("Trade Count", len(df_trades))

                # Charting
                fig = make_subplots(rows=1, cols=1)
                fig.add_trace(go.Candlestick(x=processed_df.index, open=processed_df['open'], high=processed_df['high'], 
                                             low=processed_df['low'], close=processed_df['close'], name="Price"))
                
                # Plot the Ribbon EMAs
                fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['ema20'], name="EMA 20 (Signal)", line=dict(color='yellow', width=2)))
                fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['ema30'], name="EMA 30 (Exit)", line=dict(color='orange', width=1)))
                fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['ema50'], name="EMA 50 (Baseline)", line=dict(color='red', width=2)))

                # Entry Markers
                fig.add_trace(go.Scatter(x=df_trades['entry_date'], y=df_trades['entry_price'], mode='markers', 
                                         marker=dict(symbol='triangle-up', size=12, color='lime'), name='Entry'))

                fig.update_layout(height=600, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📝 Trade Log")
                st.dataframe(df_trades[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'exit_reason', 'pnl_pct']])
    except Exception as e:
        st.error(f"Error: {e}")