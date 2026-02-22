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

# --- 2. MULTI-STRATEGY ENGINE (12 Total Strategies) ---
def run_backtest(df, symbol, config, strategy_type):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    # --- Indicator Calculations ---
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    # Bollinger Bands & Squeeze Logic
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['std_20'] = df['close'].rolling(window=20).std()
    df['upper_bb'] = df['sma_20'] + (df['std_20'] * 2)
    df['lower_bb'] = df['sma_20'] - (df['std_20'] * 2)
    df['bb_width'] = df['upper_bb'] - df['lower_bb']
    df['is_squeeze'] = df['bb_width'] <= df['bb_width'].rolling(window=20).min()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # HHV / LLV / Neckline
    h_per = config.get('hhv_period', 20)
    df['hhv'] = df['high'].rolling(window=h_per).max()
    df['llv'] = df['low'].rolling(window=h_per).min()
    df['neckline'] = df['high'].rolling(window=20).max()
    
    # Divergence
    df['price_ll'] = df['low'] < df['low'].shift(10)
    df['rsi_hl'] = df['rsi'] > df['rsi'].shift(10)
    df['price_hh'] = df['high'] > df['high'].shift(10)
    df['rsi_lh'] = df['rsi'] < df['rsi'].shift(10)

    # --- Strategy Logic Mapping ---
    if strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)

    elif strategy_type == "EMA Ribbon":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = (df['ema_fast'] < df['ema_exit']) & (df['ema_fast'].shift(1) >= df['ema_exit'].shift(1))
    
    elif strategy_type == "Bollinger Squeeze Breakout":
        # Strategy: Buy if price breaks Upper Band while in a volatility squeeze
        df['long_signal'] = df['is_squeeze'].shift(1) & (df['close'] > df['upper_bb'])
        df['exit_signal'] = (df['close'] < df['sma_20'])

    elif strategy_type == "Breakaway Gap Momentum":
        df['long_signal'] = (df['open'] > df['high'].shift(1)) & (df['close'].shift(1) >= df['hhv'].shift(2) * 0.98)
        df['exit_signal'] = (df['close'] < df['low'].shift(1))

    elif strategy_type == "EMA & RSI Synergy":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['rsi'] > 60)
        df['exit_signal'] = (df['close'] < df['ema_exit']) | (df['rsi'] < 40)

    elif strategy_type == "RSI Divergence":
        df['long_signal'] = df['price_ll'] & df['rsi_hl'] & (df['close'] > df['high'].shift(1))
        df['exit_signal'] = df['price_hh'] & df['rsi_lh']

    elif strategy_type == "BB & RSI Exhaustion":
        df['long_signal'] = (df['low'] <= df['lower_bb']) & (df['rsi'] < 30)
        df['exit_signal'] = (df['close'] >= df['sma_20']) | (df['rsi'] > 50)

    elif strategy_type == "ATR Band Breakout":
        df['upper_atr'] = df['sma_20'] + df['atr']
        df['lower_atr'] = df['sma_20'] - df['atr']
        df['long_signal'] = (df['close'] > df['upper_atr']) & (df['close'].shift(1) <= df['upper_atr'].shift(1))
        df['exit_signal'] = (df['close'] < df['lower_atr']) & (df['close'].shift(1) >= df['lower_atr'].shift(1))
        
    elif strategy_type == "HHV/LLV Breakout":
        df['long_signal'] = (df['close'] > df['hhv'].shift(1))
        df['exit_signal'] = (df['close'] < df['llv'].shift(1))

    elif strategy_type == "Double Bottom Breakout":
        df['long_signal'] = (df['close'] > df['neckline'].shift(1))
        df['exit_signal'] = (df['close'] < df['ema_exit'])

    elif strategy_type == "Fibonacci 61.8% Retracement":
        uptrend = df['close'] > df['sma_200']
        fib_lvl = df['hhv'] - ((df['hhv'] - df['llv']) * 0.618)
        df['long_signal'] = uptrend & (df['low'] <= fib_lvl) & (df['close'] > df['high'].shift(1))
        df['exit_signal'] = df['close'] < df['llv'].shift(1)

    elif strategy_type == "Relative Strength Play":
        df['stock_ret'] = df['close'].pct_change(periods=55)
        df['long_signal'] = (df['stock_ret'] > 0) & (df['close'] > df['ema_fast'])
        df['exit_signal'] = (df['close'] < df['ema_slow'])

    # --- Backtest Loop ---
    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            if sl_hit or tp_hit or prev['exit_signal']:
                reason = "Stop Loss" if sl_hit else ("Target Profit" if tp_hit else "Signal Exit")
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
    return trades, df

# --- 3. UI STYLING ---
st.set_page_config(layout="wide", page_title="Strategy Lab Pro")
st.markdown("""
    <style>
    .stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; }
    .report-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .report-table td { border: 1px solid #2d2f3b; padding: 10px; text-align: center; color: #fff; font-size: 0.85rem; }
    .profit { background-color: #1b5e20 !important; color: #c8e6c9 !important; font-weight: bold; }
    .loss { background-color: #b71c1c !important; color: #ffcdd2 !important; font-weight: bold; }
    .total-cell { font-weight: bold; color: #fff; background-color: #1e3a5f !important; }
    .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; }
    .stat-label { color: #999; font-size: 0.85rem; }
    .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strat_choice = st.sidebar.selectbox("Select Strategy", ["RSI 60 Cross", "EMA Ribbon", "Bollinger Squeeze Breakout", "Breakaway Gap Momentum", "EMA & RSI Synergy", "RSI Divergence", "BB & RSI Exhaustion", "Relative Strength Play", "ATR Band Breakout", "HHV/LLV Breakout", "Double Bottom Breakout", "Fibonacci 61.8% Retracement"])
tf_map = {"1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", "1 Hour": "1h", "Daily": "1d"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=4)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2005-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 30, 'hhv_period': 20}
st.sidebar.divider()
use_sl = st.sidebar.toggle("Stop Loss", True); config['sl_val'] = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0; config['use_sl'] = use_sl
use_tp = st.sidebar.toggle("Target Profit", True); config['tp_val'] = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0; config['use_tp'] = use_tp
use_slip = st.sidebar.toggle("Slippage", True); config['slippage_val'] = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slip else 0; config['use_slippage'] = use_slip

# --- 5. EXECUTION ---
if st.sidebar.button("🚀 Run Backtest"):
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            trades, processed_df = run_backtest(data, symbol, config, strat_choice)
            
            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
                df_trades['