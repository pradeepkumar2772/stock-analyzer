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
    direction: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime = None
    exit_price: float = None
    exit_reason: str = None
    pnl_pct: float = 0.0

# --- 2. RS–AV CALCULATION ---
def calculate_rsav(stock_df, benchmark_df, lookback=50):
    # Stock Risk-Adjusted Alpha
    s_roc = stock_df['close'].pct_change(lookback) * 100
    s_vol = s_roc.rolling(window=lookback).std()
    s_net = s_roc - s_vol
    
    # Benchmark Risk-Adjusted Alpha
    b_roc = benchmark_df['close'].pct_change(lookback) * 100
    b_vol = b_roc.rolling(window=lookback).std()
    b_net = b_roc - b_vol
    
    return s_net - b_net

# --- 3. MULTI-STRATEGY ENGINE ---
def run_backtest(df, benchmark_df, symbol, config, strategy_type):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    # Pre-calculate RS–AV
    df['rsav'] = calculate_rsav(df, benchmark_df, lookback=config['rsav_lookback'])
    
    # Indicators
    df['ema_fast'] = df['close'].ewm(span=config['ema_fast'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config['ema_slow'], adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config['ema_exit'], adjust=False).mean()

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        # RS–AV Trigger Condition
        market_ok = True if not config['use_rsav'] else curr['rsav'] >= config['rsav_trigger']
        
        if active_trade:
            if prev['ema_fast'] < prev['ema_exit']:
                active_trade.exit_price = curr['open'] * (1 - slippage)
                active_trade.exit_date = curr.name
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
                
        elif prev['ema_fast'] > prev['ema_slow'] and market_ok:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=curr.name, entry_price=curr['open'] * (1 + slippage))
            
    return trades, df

# --- 4. UI STYLING ---
st.set_page_config(layout="wide", page_title="Pro-Tracer RS-Alpha")
st.markdown("<style>.stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; } .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; } .stat-label { color: #999; font-size: 0.85rem; } .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }</style>", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- 5. SIDEBAR ---
st.sidebar.title("🎗️ Pro-Tracer Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
capital = st.sidebar.number_input("Initial Capital", value=100000.0)
start_str = st.sidebar.text_input("Start Date", value="2018-01-01")

st.sidebar.divider()
st.sidebar.subheader("🛡️ Market Filter (RS–AV)")
use_rsav = st.sidebar.toggle("Enable RS–AV Filter", True)
rsav_trigger = st.sidebar.number_input("Trigger Level", value=-0.5, step=0.1)
rsav_lookback = st.sidebar.selectbox("Look-back Period", [50, 100, 252], index=0)

# Fetch Data
@st.cache_data
def fetch_data(sym, start):
    s = yf.download(sym, start=start, auto_adjust=True)
    b = yf.download("^NSEI", start=start, auto_adjust=True)
    for d in [s, b]:
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d.columns = [str(col).lower() for col in d.columns]
    idx = s.index.intersection(b.index)
    return s.loc[idx], b.loc[idx]

s_data, b_data = fetch_data(symbol, start_str)

# --- 6. EXECUTION ---
if st.sidebar.button("🚀 Run Backtest"):
    config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 15, 'use_rsav': use_rsav, 
              'rsav_trigger': rsav_trigger, 'rsav_lookback': rsav_lookback,
              'use_slippage': True, 'slippage_val': 0.1, 'use_sl': False, 'use_tp': False}
    
    trades, processed_df = run_backtest(s_data, b_data, symbol, config, "EMA Ribbon")
    
    if trades:
        df_t = pd.DataFrame([vars(t) for t in trades])
        df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
        df_t['year'] = pd.to_datetime(df_t['exit_date']).dt.year
        
        # Metrics Calculations
        wins = df_t[df_t['pnl_pct'] > 0]; losses = df_t[df_t['pnl_pct'] <= 0]
        total_ret = (df_t['equity'].iloc[-1] / capital - 1) * 100
        peak = df_t['equity'].cummax(); drawdown = (df_t['equity'] - peak) / peak; mdd = drawdown.min() * 100
        
        t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Relative Strength", "Trade Log"])
        
        with t1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Return", f"{total_ret:.2f}%"); c2.metric("Max DD", f"{mdd:.2f}%"); c3.metric("Win Rate", f"{(len(wins)/len(df_t)*100):.2f}%"); c4.metric("Trades", len(df_t))
            st.plotly_chart(px.line(df_t, x='exit_date', y='equity', title="Capital Growth"), use_container_width=True)

        with t2:
            cl, cr = st.columns([1, 2.5])
            with cl:
                with st.expander("📊 Performance", expanded=True): draw_stat("Total Return", f"{total_ret:.2f}%")
                with st.expander("📉 Drawdown"): draw_stat("Max DD", f"{mdd:.2f}%")
                with st.expander("🏆 Performance"): draw_stat("Win Rate", f"{(len(wins)/len(df_t)*100):.2f}%")
                with st.expander("🔍 Characteristics"): draw_stat("Total Trades", len(df_t))
            with cr:
                st.plotly_chart(px.area(df_t, x='exit_date', y=drawdown*100, title="Underwater Drawdown (%)", color_discrete_sequence=['#e74c3c']), use_container_width=True)

        with t3:
            st.subheader("RS–Alpha Volatility Index")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['rsav'], fill='tozeroy', name="RS-AV Alpha"))
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            fig.add_hline(y=rsav_trigger, line_dash="dot", line_color="orange", annotation_text="Trigger")
            st.plotly_chart(fig, use_container_width=True)

        with t4: st.dataframe(df_t, use_container_width=True)
    else: st.warning("No trades found.")