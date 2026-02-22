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
    
    # EMA Calculations
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Logic
    df['long_signal'] = (df['ema20'] > df['ema50']) & (df['ema20'].shift(1) <= df['ema50'].shift(1))
    df['exit_signal'] = (df['ema20'] < df['ema30']) & (df['ema20'].shift(1) >= df['ema30'].shift(1))

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        if active_trade:
            sl_hit = False
            tp_hit = False
            
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
st.set_page_config(layout="wide", page_title="Full History Backtester")

st.sidebar.title("🎗️ PK Ribbon Engine")
symbol = st.sidebar.text_input("Symbol", value="^GSPC") # Defaulted to S&P 500 for testing long history
capital = st.sidebar.number_input("Initial Capital", value=100000)

# DATE RANGE (Set to 50 years max)
fifty_years_ago = date.today() - timedelta(days=50*365)
start_date = st.sidebar.date_input("Start Date", value=date(1980, 1, 1), min_value=fifty_years_ago)
end_date = st.sidebar.date_input("End Date", value=date.today())

st.sidebar.divider()
st.sidebar.subheader("⚙️ Toggles")
use_sl = st.sidebar.checkbox("Enable Stop Loss", value=True)
sl_val = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0

use_tp = st.sidebar.checkbox("Enable Target Profit", value=True)
tp_val = st.sidebar.slider("Target %", 1.0, 100.0, 25.0) if use_tp else 0

use_slippage = st.sidebar.checkbox("Apply Slippage", value=True)
slippage_val = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slippage else 0

if st.sidebar.button("🚀 Run Full History Backtest"):
    try:
        with st.spinner('Downloading and Processing Global Data...'):
            # Fetching data without any 'tail' or 'period' limits
            data = yf.download(symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True)
            
            if data.empty:
                st.error("No data found for this symbol/date range.")
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data.columns = [str(col).lower() for col in data.columns]
                data = data.dropna()
                
                # Report actual range found
                st.success(f"Loaded {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")
                
                config = {
                    'use_sl': use_sl, 'sl_val': sl_val, 'use_tp': use_tp, 'tp_val': tp_val,
                    'use_slippage': use_slippage, 'slippage_val': slippage_val, 'capital': capital
                }
                
                trades, processed_df = run_backtest(data, symbol, config)

                if not trades:
                    st.warning("Strategy generated 0 trades. Try adjusting parameters.")
                else:
                    df_trades = pd.DataFrame([vars(t) for t in trades])
                    total_ret = (df_trades['pnl_pct'] + 1).prod() - 1
                    win_rate = (len(df_trades[df_trades['pnl_pct'] > 0]) / len(df_trades)) * 100

                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Return", f"{total_ret*100:.1f}%")
                    m2.metric("Win Rate", f"{win_rate:.1f}%")
                    m3.metric("Trades", len(df_trades))
                    m4.metric("Final Value", f"₹{capital * (1+total_ret):,.0f}")

                    # UNLIMITED CHARTING
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    
                    # Candlestick - NO .tail() LIMIT
                    fig.add_trace(go.Candlestick(x=processed_df.index, open=processed_df['open'], high=processed_df['high'], 
                                                 low=processed_df['low'], close=processed_df['close'], name="Price"), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['ema20'], name="EMA 20", line=dict(color='yellow', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['ema50'], name="EMA 50", line=dict(color='red', width=1)), row=1, col=1)

                    # Full Equity Curve
                    df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                    fig.add_trace(go.Scatter(x=df_trades['exit_date'], y=df_trades['equity'], name="Equity Curve", line=dict(color='#00ffcc')), row=2, col=1)

                    fig.update_layout(height=1000, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📜 Complete Trade Log")
                    st.dataframe(df_trades)
    except Exception as e:
        st.error(f"Error: {e}")