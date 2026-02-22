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
    # Stock Risk-Adjusted Alpha (Return - Volatility)
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
    
    # Pre-calculate RS-AV Alpha
    df['rsav'] = calculate_rsav(df, benchmark_df, lookback=config.get('rsav_lookback', 50))
    
    # Indicators
    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()

    # ATR for Trailing Stop
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    highest_high = 0

    # --- Strategy Signals ---
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20'].shift(1)) & (df['close'] > df['ema_20'])
        df['exit_signal'] = (df['close'] < df['ema_15'])
    elif strategy_type == "EMA Ribbon":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = (df['ema_fast'] < df['ema_exit'])

    # --- Backtest Loop ---
    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        
        # RS-AV Market Filter Logic
        market_ok = True if not config['use_rsav'] else current['rsav'] >= config['rsav_trigger']
        
        if active_trade:
            highest_high = max(highest_high, current['high'])
            trailing_stop = highest_high - (current['atr'] * config.get('atr_mult', 3.0))
            ts_hit = config.get('use_ts', False) and current['low'] <= trailing_stop
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            
            if sl_hit or ts_hit or prev['exit_signal']:
                active_trade.exit_price = min(current['open'], trailing_stop) if ts_hit else current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = "TS Hit" if ts_hit else ("SL" if sl_hit else "Signal")
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None; highest_high = 0
        elif prev['long_signal'] and market_ok:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
            highest_high = current['high']
    return trades, df

# --- 4. UI STYLING & SIDEBAR ---
st.set_page_config(layout="wide", page_title="Pro-Tracer Pro")
st.markdown("<style>.stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; } .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; } .stat-label { color: #999; font-size: 0.85rem; } .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }</style>", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

st.sidebar.title("🎗️ Pro-Tracer Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strat_choice = st.sidebar.selectbox("Select Strategy", ["PK Strategy (Positional)", "EMA Ribbon"])
capital = st.sidebar.number_input("Initial Capital", value=100000.0)
start_str = st.sidebar.text_input("Start Date", value="2018-01-01")

# Market Filter Controls
st.sidebar.divider()
st.sidebar.subheader("🛡️ Market Filter (RS–AV)")
use_rsav = st.sidebar.toggle("Enable RS–AV Filter", True)
rsav_trigger = st.sidebar.number_input("Trigger Level", value=-0.5, step=0.1)

# Backtest Controls
st.sidebar.divider()
use_ts = st.sidebar.toggle("ATR Trailing Stop", False); config_ts = st.sidebar.slider("ATR Multiplier", 1.0, 5.0, 3.0) if use_ts else 3.0
use_sl = st.sidebar.toggle("Stop Loss", True); sl_v = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0

# --- 5. DATA FETCHING ---
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
              'rsav_trigger': rsav_trigger, 'rsav_lookback': 50, 'use_ts': use_ts, 
              'atr_mult': config_ts, 'use_sl': use_sl, 'sl_val': sl_v, 'use_slippage': True, 'slippage_val': 0.1}
    
    trades, processed_df = run_backtest(s_data, b_data, symbol, config, strat_choice)
    
    if trades:
        df_t = pd.DataFrame([vars(t) for t in trades])
        df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
        total_ret = (df_t['equity'].iloc[-1] / capital - 1) * 100
        peak = df_t['equity'].cummax(); drawdown = (df_t['equity'] - peak) / peak * 100
        
        t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
        
        with t1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Return", f"{total_ret:.2f}%"); c2.metric("Max DD", f"{drawdown.min():.2f}%"); c3.metric("Trades", len(df_t))
            st.plotly_chart(px.line(df_t, x='exit_date', y='equity', title="Capital Growth"), use_container_width=True)

        with t2:
            cl, cr = st.columns([1, 2.5])
            with cl:
                with st.expander("📊 Performance", expanded=True): draw_stat("Total Ret", f"{total_ret:.2f}%")
                with st.expander("⏱️ Holding Period"): 
                    df_t['hold'] = (pd.to_datetime(df_t['exit_date']) - pd.to_datetime(df_t['entry_date'])).dt.days
                    draw_stat("Avg Hold", f"{df_t['hold'].mean():.2f} days")
            with cr:
                st.plotly_chart(px.area(df_t, x='exit_date', y=drawdown, title="Drawdown (%)", color_discrete_sequence=['#e74c3c']), use_container_width=True)

        with t4: st.dataframe(df_t, use_container_width=True)
    else: st.warning("No trades found.")