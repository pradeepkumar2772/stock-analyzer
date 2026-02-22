import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date

# --- 1. CORE DATA STRUCTURE ---
@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime = None
    exit_price: float = None
    exit_reason: str = None
    pnl_pct: float = 0.0

# --- 2. RS–AV CALCULATION ENGINE ---
def calculate_rsav(stock_df, benchmark_df, lookback=50):
    # Stock ROC and Volatility
    stock_roc = stock_df['close'].pct_change(lookback) * 100
    stock_vol = stock_roc.rolling(window=lookback).std()
    stock_net = stock_roc - stock_vol
    
    # Benchmark ROC and Volatility
    bench_roc = benchmark_df['close'].pct_change(lookback) * 100
    bench_vol = bench_roc.rolling(window=lookback).std()
    bench_net = bench_roc - bench_vol
    
    return stock_net - bench_net

# --- 3. UPDATED BACKTEST ENGINE ---
def run_backtest(df, benchmark_df, symbol, config, strategy_type):
    trades = []
    active_trade = None
    
    # Pre-calculate RS–AV
    df['rsav'] = calculate_rsav(df, benchmark_df)
    
    # EMAs for Signals
    df['ema_fast'] = df['close'].ewm(span=config['ema_fast'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config['ema_slow'], adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config['ema_exit'], adjust=False).mean()

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Market Filter Condition
        market_ok = True if not config['use_rsav'] else curr['rsav'] > 0
        
        if active_trade:
            # Exit Logic
            if prev['ema_fast'] < prev['ema_exit']:
                active_trade.exit_price = curr['open']
                active_trade.exit_date = curr.name
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None
                
        elif prev['ema_fast'] > prev['ema_slow'] and market_ok:
            active_trade = Trade(symbol=symbol, entry_date=curr.name, entry_price=curr['open'])
            
    return trades, df

# --- 4. UI & SIDEBAR ---
st.set_page_config(layout="wide", page_title="Pro-Tracer RS-Alpha")
st.sidebar.title("🎗️ Pro-Tracer Engine")

symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
capital = st.sidebar.number_input("Capital", value=100000.0)
start_date = st.sidebar.text_input("Start Date", value="2018-01-01")

st.sidebar.divider()
st.sidebar.markdown("### Market Filters")
use_rsav = st.sidebar.toggle("Enable RS–AV Filter", help="Only trade if stock outperformance > Nifty 50 Alpha")

# Data Download
@st.cache_data
def fetch_all_data(sym, start):
    s_data = yf.download(sym, start=start, auto_adjust=True)
    b_data = yf.download("^NSEI", start=start, auto_adjust=True) # Nifty 50
    
    for d in [s_data, b_data]:
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d.columns = [str(col).lower() for col in d.columns]
        
    # Sync dates
    common_idx = s_data.index.intersection(b_data.index)
    return s_data.loc[common_idx], b_data.loc[common_idx]

stock_data, nifty_data = fetch_all_data(symbol, start_date)

if st.sidebar.button("🚀 Run Analysis"):
    config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 15, 'use_rsav': use_rsav}
    trades, processed_df = run_backtest(stock_data, nifty_data, symbol, config, "EMA Ribbon")
    
    if trades:
        df_t = pd.DataFrame([vars(t) for t in trades])
        df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
        
        t1, t2, t3 = st.tabs(["Performance", "Relative Strength", "Trade Log"])
        
        with t1:
            st.metric("Total Return", f"{(df_t['equity'].iloc[-1]/capital - 1)*100:.2f}%")
            st.plotly_chart(px.line(df_t, x='exit_date', y='equity', title="Capital Growth"), use_container_width=True)
            
        with t2:
            st.subheader("RS–Alpha Volatility Index")
            
            fig_rsav = go.Figure()
            fig_rsav.add_trace(go.Scatter(x=processed_df.index, y=processed_df['rsav'], 
                                          name="RS-AV", fill='tozeroy',
                                          line=dict(color='lime' if processed_df['rsav'].iloc[-1] > 0 else 'red')))
            fig_rsav.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig_rsav, use_container_width=True)
            
        with t3:
            st.dataframe(df_t, use_container_width=True)