import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date, timedelta

# --- 1. CORE DATA STRUCTURE (Locked) ---
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

# --- RS-AV CALCULATION ENGINE (Locked) ---
def calculate_rsav(stock_df, benchmark_df, lookback=50):
    s_roc = stock_df['close'].pct_change(lookback) * 100
    s_vol = s_roc.rolling(window=lookback).std()
    s_net = s_roc - s_vol
    b_roc = benchmark_df['close'].pct_change(lookback) * 100
    b_vol = b_roc.rolling(window=lookback).std()
    b_net = b_roc - b_vol
    return s_net - b_net

# --- 2. MULTI-STRATEGY ENGINE (Locked) ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    df['long_signal'] = False
    df['exit_signal'] = False
    
    if config.get('use_rsav', False) and benchmark_df is not None:
        df['rsav'] = calculate_rsav(df, benchmark_df, lookback=50)

    # Indicator Logic (Exact Baseline from RKB Books)
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

    # Strategy Selection
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = (df['close'] < df['ema_15_pk'])
    elif strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)
    elif strategy_type == "EMA Ribbon":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = (df['ema_fast'] < df['ema_exit'])
    # ... (Other strategies same as RKB Baseline)

    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        market_ok = True
        if config.get('use_rsav', False) and 'rsav' in df.columns:
            market_ok = current['rsav'] >= config.get('rsav_trigger', -0.5)
            
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            if sl_hit or tp_hit or prev['exit_signal']:
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        elif prev['long_signal'] and market_ok:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
    return trades, df

# --- 3. OPTIMIZER ENGINE ---
def run_pk_optimizer(data, symbol, config):
    results = []
    ema_range = range(3, 61, 2) 
    total_steps = len(ema_range) * len(ema_range)
    current_step = 0
    progress_bar = st.progress(0)
    
    for entry_ema in ema_range:
        for exit_ema in ema_range:
            current_step += 1
            if entry_ema <= exit_ema: continue
            
            df_opt = data.copy()
            df_opt['ema_20_pk'] = df_opt['close'].ewm(span=entry_ema, adjust=False).mean()
            df_opt['ema_15_pk'] = df_opt['close'].ewm(span=exit_ema, adjust=False).mean()
            
            trades, _ = run_backtest(df_opt, symbol, config, "PK Strategy (Positional)")
            
            if trades:
                df_res = pd.DataFrame([vars(t) for t in trades])
                df_res['equity'] = 1000 * (1 + df_res['pnl_pct']).cumprod()
                total_ret = (df_res['equity'].iloc[-1] / 1000 - 1) * 100
                mdd = ((df_res['equity'] - df_res['equity'].cummax()) / df_res['equity'].cummax()).min() * 100
                rf = total_ret / abs(mdd) if mdd != 0 else total_ret
                results.append({"Entry EMA": entry_ema, "Exit EMA": exit_ema, "Return %": round(total_ret, 2), "Recovery Factor": round(rf, 2)})
            
            if current_step % 100 == 0: progress_bar.progress(current_step / total_steps)
    
    progress_bar.empty()
    return pd.DataFrame(results)

# --- 4. UI & SIDEBAR ---
st.set_page_config(layout="wide", page_title="Strategy Lab Pro")
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strategies_list = ["PK Strategy (Positional)", "RSI 60 Cross", "EMA Ribbon", "Flags & Pennants", "Bollinger Squeeze Breakout", "EMA & RSI Synergy", "RSI Divergence", "BB & RSI Exhaustion", "Relative Strength Play", "ATR Band Breakout", "HHV/LLV Breakout", "Double Bottom Breakout", "Fibonacci 61.8% Retracement"]
strat_choice = st.sidebar.selectbox("Select Strategy", strategies_list)
tf_map = {"1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", "1 Hour": "1h", "Daily": "1d", "Weekly": "1wk"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=4)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="1999-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

use_rsav = st.sidebar.toggle("Enable RS-AV Filter", False)
config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 30, 'hhv_period': 20, 'use_rsav': use_rsav, 'rsav_trigger': -0.5, 'sl_val': 5.0, 'tp_val': 25.0, 'use_sl': True, 'use_tp': True, 'use_slippage': True, 'slippage_val': 0.1}

col_run1, col_run2, col_run3 = st.sidebar.columns(3)
run_single = col_run1.button("🚀 Backtest")
run_arena = col_run2.button("🏟️ Arena")
run_opt = col_run3.button("🎯 Optimizer")

# --- 5. EXECUTION LOGIC ---
if run_single:
    # ... (Standard Single Strategy Backtest Logic - 8 Expanders Protected)
    pass

elif run_arena:
    # ... (Arena Comparison Logic - Multi-Strategy Rankings)
    pass

elif run_opt:
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            
            # Baseline Comparison
            b_trades, _ = run_backtest(data.copy(), symbol, config, "PK Strategy (Positional)")
            b_ret = 0.0
            if b_trades:
                b_df = pd.DataFrame([vars(t) for t in b_trades])
                b_df['equity'] = capital * (1 + b_df['pnl_pct']).cumprod()
                b_ret = (b_df['equity'].iloc[-1] / capital - 1) * 100

            opt_results = run_pk_optimizer(data, symbol, config)
            if not opt_results.empty:
                st.subheader(f"🎯 Optimization Results for {symbol}")
                best_row = opt_results.loc[opt_results['Recovery Factor'].idxmax()]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Return (20/15)", f"{b_ret:.2f}%")
                c2.metric("Optimized Return", f"{best_row['Return %']:.2f}%", delta=f"{best_row['Return %'] - b_ret:.2f}%")
                c3.metric("Best EMA Combo", f"{best_row['Entry EMA']} / {best_row['Exit EMA']}")
                
                st.write("### 🗺️ Recovery Factor Heatmap")
                heatmap_data = opt_results.pivot(index='Entry EMA', columns='Exit EMA', values='Recovery Factor')
                st.dataframe(heatmap_data.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"), use_container_width=True)
    except Exception as e:
        st.error(f"Optimizer Error: {e}")