import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date, timedelta

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

# --- 2. RS–AV CALCULATION ENGINE ---
def calculate_rsav(stock_df, benchmark_df, lookback=50):
    s_roc = stock_df['close'].pct_change(lookback) * 100
    s_vol = s_roc.rolling(window=lookback).std()
    s_net = s_roc - s_vol
    
    b_roc = benchmark_df['close'].pct_change(lookback) * 100
    b_vol = b_roc.rolling(window=lookback).std()
    b_net = b_roc - b_vol
    return s_net - b_net

# --- 3. MULTI-STRATEGY ENGINE ---
def run_backtest(df, benchmark_df, symbol, config, strategy_type):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    # Indicators
    df['rsav'] = calculate_rsav(df, benchmark_df, lookback=50)
    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()
    
    # ATR for Trailing Stop
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    highest_high = 0

    for i in range(1, len(df)):
        curr = df.iloc[i]; prev = df.iloc[i-1]
        market_ok = True if not config['use_rsav'] else curr['rsav'] >= config['rsav_trigger']
        
        if active_trade:
            highest_high = max(highest_high, curr['high'])
            trailing_stop = highest_high - (curr['atr'] * config.get('atr_mult', 3.0))
            ts_hit = config.get('use_ts', False) and curr['low'] <= trailing_stop
            sl_hit = config['use_sl'] and curr['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            
            if sl_hit or ts_hit or (strategy_type == "PK Strategy (Positional)" and curr['close'] < curr['ema_15']) or (strategy_type != "PK Strategy (Positional)" and prev['ema_fast'] < prev['ema_exit']):
                active_trade.exit_price = min(curr['open'], trailing_stop) if ts_hit else curr['open'] * (1 - slippage)
                active_trade.exit_date = curr.name
                active_trade.exit_reason = "TS Hit" if ts_hit else "Signal/SL"
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None; highest_high = 0
        elif market_ok:
            if strategy_type == "PK Strategy (Positional)" and prev['close'] < prev['ema_20'] and curr['close'] > curr['ema_20']:
                active_trade = Trade(symbol=symbol, direction="Long", entry_date=curr.name, entry_price=curr['open'] * (1 + slippage))
                highest_high = curr['high']
            elif strategy_type == "EMA Ribbon" and prev['ema_fast'] > prev['ema_slow']:
                active_trade = Trade(symbol=symbol, direction="Long", entry_date=curr.name, entry_price=curr['open'] * (1 + slippage))
                highest_high = curr['high']
    return trades, df

# --- 4. UI STYLING ---
st.set_page_config(layout="wide", page_title="Pro-Tracer Pro")
st.markdown("<style>.stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; } .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; } .stat-label { color: #999; font-size: 0.85rem; } .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }</style>", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- 5. SIDEBAR ---
st.sidebar.title("🎗️ Pro-Tracer Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strat_choice = st.sidebar.selectbox("Select Strategy", ["PK Strategy (Positional)", "EMA Ribbon"])
capital = st.sidebar.number_input("Initial Capital", value=100000.0)
start_str = st.sidebar.text_input("Start Date", value="2018-01-01")

st.sidebar.divider()
use_rsav = st.sidebar.toggle("Enable RS–AV Filter", True)
rsav_trig = st.sidebar.number_input("RS-AV Trigger", value=-0.5)

st.sidebar.divider()
use_ts = st.sidebar.toggle("ATR Trailing Stop", False); atr_m = st.sidebar.slider("ATR Mult", 1.0, 5.0, 3.0)
use_sl = st.sidebar.toggle("Stop Loss", True); sl_v = st.sidebar.slider("SL %", 0.5, 15.0, 5.0)
use_slip = st.sidebar.toggle("Slippage", True); slip_v = 0.1 if use_slip else 0

# --- 6. EXECUTION ---
if st.sidebar.button("🚀 Run Backtest"):
    s = yf.download(symbol, start=start_str, auto_adjust=True)
    b = yf.download("^NSEI", start=start_str, auto_adjust=True)
    for d in [s, b]:
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d.columns = [str(col).lower() for col in d.columns]
    idx = s.index.intersection(b.index)
    s_data, b_data = s.loc[idx], b.loc[idx]
    
    config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 15, 'use_rsav': use_rsav, 'rsav_trigger': rsav_trig, 'use_ts': use_ts, 'atr_mult': atr_m, 'use_sl': use_sl, 'sl_val': sl_v, 'use_slippage': use_slip, 'slippage_val': slip_v}
    trades, processed_df = run_backtest(s_data, b_data, symbol, config, strat_choice)
    
    if trades:
        df_t = pd.DataFrame([vars(t) for t in trades])
        df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
        wins = df_t[df_t['pnl_pct'] > 0]
        total_ret = (df_t['equity'].iloc[-1] / capital - 1) * 100
        mdd = ((df_t['equity'] - df_t['equity'].cummax()) / df_t['equity'].cummax()).min() * 100

        t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
        with t1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Return", f"{total_ret:.2f}%"); c2.metric("Max DD", f"{mdd:.2f}%"); c3.metric("Win Ratio", f"{(len(wins)/len(df_t)*100):.2f}%"); c4.metric("Trades", len(df_t))
            st.plotly_chart(px.line(df_t, x='exit_date', y='equity', title="Capital Growth"), use_container_width=True)
        with t2:
            cl, cr = st.columns([1, 2.5])
            with cl:
                with st.expander("📊 Performance", expanded=True): draw_stat("Total Ret", f"{total_ret:.2f}%")
                with st.expander("⏱️ Holding Period"): 
                    df_t['hold'] = (pd.to_datetime(df_t['exit_date']) - pd.to_datetime(df_t['entry_date'])).dt.days
                    draw_stat("Avg Hold", f"{df_t['hold'].mean():.2f} days")
            with cr:
                st.plotly_chart(px.area(df_t, x='exit_date', y=(df_t['equity'] - df_t['equity'].cummax()) / df_t['equity'].cummax() * 100, title="Drawdown (%)", color_discrete_sequence=['#e74c3c']), use_container_width=True)
        with t4: st.dataframe(df_t, use_container_width=True)