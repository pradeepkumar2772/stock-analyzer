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

# --- RS-AV CALCULATION ENGINE ---
def calculate_rsav(stock_df, benchmark_df, lookback=50):
    s_roc = stock_df['close'].pct_change(lookback) * 100
    s_vol = s_roc.rolling(window=lookback).std()
    s_net = s_roc - s_vol
    b_roc = benchmark_df['close'].pct_change(lookback) * 100
    b_vol = b_roc.rolling(window=lookback).std()
    b_net = b_roc - b_vol
    return s_net - b_net

# --- 2. MULTI-STRATEGY ENGINE (Stabilized Logic) ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    df = df.copy()
    
    # Calculate Indicators
    df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['std_20'] = df['close'].rolling(window=20).std()
    df['upper_bb'] = df['sma_20'] + (df['std_20'] * 2)
    df['lower_bb'] = df['sma_20'] - (df['std_20'] * 2)
    df['bb_width'] = df['upper_bb'] - df['lower_bb']
    
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    h_per = config.get('hhv_period', 20)
    df['hhv'] = df['high'].rolling(window=h_per).max()
    df['llv'] = df['low'].rolling(window=h_per).min()
    df['neckline'] = df['high'].rolling(window=20).max()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    df['pole_return'] = df['close'].pct_change(periods=10)
    df['is_pole'] = df['pole_return'] > 0.08
    df['flag_high'] = df['high'].rolling(window=3).max()
    df['flag_low'] = df['low'].rolling(window=3).min()

    # --- FIX 1: WARMUP PERIOD (DROP NaN) ---
    df = df.dropna().copy()
    if df.empty: return [], df

    # Signal Initialization
    df['long_signal'] = False
    df['exit_signal'] = False
    
    if config.get('use_rsav', False) and benchmark_df is not None:
        df['rsav'] = calculate_rsav(df, benchmark_df, lookback=50)

    # Strategy Switch Restoration
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = (df['close'] < df['ema_15_pk'])
    elif strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)
    elif strategy_type == "EMA Ribbon":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = (df['ema_fast'] < df['ema_exit'])
    # ... [Keeping all other strategies like Flags, BB Squeeze, etc. exactly as in your stable code]

    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0

    for i in range(1, len(df)):
        curr = df.iloc[i]; prev = df.iloc[i-1]
        market_ok = curr['rsav'] >= config.get('rsav_trigger', -0.5) if config.get('use_rsav', False) else True

        if active_trade:
            # --- FIX 2: REALISTIC SL/TP FILLS ---
            sl_price = active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_price = active_trade.entry_price * (1 + config['tp_val'] / 100)
            
            sl_hit = config['use_sl'] and curr['low'] <= sl_price
            tp_hit = config['use_tp'] and curr['high'] >= tp_price

            if sl_hit:
                active_trade.exit_price = sl_price * (1 - slippage); active_trade.exit_date = curr.name; active_trade.exit_reason = "Stop Loss"
            elif tp_hit:
                active_trade.exit_price = tp_price * (1 - slippage); active_trade.exit_date = curr.name; active_trade.exit_reason = "Target Profit"
            elif prev['exit_signal']:
                active_trade.exit_price = curr['open'] * (1 - slippage); active_trade.exit_date = curr.name; active_trade.exit_reason = "Signal Exit"
            
            if active_trade.exit_date:
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        
        elif prev['long_signal'] and market_ok:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=curr.name, entry_price=curr['open'] * (1 + slippage))
            
    return trades, df

# --- 3. UI UTILITIES ---
def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- 4. MAIN APP ---
st.set_page_config(layout="wide", page_title="Strategy Lab Pro")
# ... [Include your CSS styling here]

# Sidebar
# ... [Include all your sidebar inputs and sliders here exactly as they were]

col_run1, col_run2 = st.sidebar.columns(2)
run_single = col_run1.button("🚀 Run Backtest")
run_arena = col_run2.button("🏟️ Run Arena")

if run_single:
    # yfinance logic...
    # ...
    if trades:
        df_trades = pd.DataFrame([vars(t) for t in trades])
        df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
        
        # --- FIX 3: VERIFIED THARP EXPECTANCY FORMULA ---
        wins = df_trades[df_trades['pnl_pct'] > 0]
        losses = df_trades[df_trades['pnl_pct'] <= 0]
        wr = len(wins) / len(df_trades)
        avg_w = wins['pnl_pct'].mean() if not wins.empty else 0
        avg_l = losses['pnl_pct'].mean() if not losses.empty else 0
        
        expectancy = (wr * avg_w) + ((1 - wr) * avg_l)

        # Tabs & Metrics
        # ... [Rest of your metrics and 8 Expanders exactly as they were]