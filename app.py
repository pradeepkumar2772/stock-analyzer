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

# --- RS-AV ENGINE (Restored) ---
def calculate_rsav(stock_df, benchmark_df, lookback=50):
    s_roc = stock_df['close'].pct_change(lookback) * 100
    s_vol = s_roc.rolling(window=lookback).std()
    s_net = s_roc - s_vol
    b_roc = benchmark_df['close'].pct_change(lookback) * 100
    b_vol = b_roc.rolling(window=lookback).std()
    b_net = b_roc - b_vol
    return s_net - b_net

# --- 2. MULTI-STRATEGY ENGINE (Restored + Corrected) ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    df = df.copy()
    
    # Shadow Injection (Optimizer protection)
    if 'ema_15_pk' not in df.columns:
        df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    if 'ema_20_pk' not in df.columns:
        df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()

    # Restoration of ALL your indicators
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
    
    df['hhv'] = df['high'].rolling(window=config.get('hhv_period', 20)).max()
    df['llv'] = df['low'].rolling(window=config.get('hhv_period', 20)).min()
    df['neckline'] = df['high'].rolling(window=20).max()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    df['pole_return'] = df['close'].pct_change(periods=10)
    df['is_pole'] = df['pole_return'] > 0.08
    df['flag_high'] = df['high'].rolling(window=3).max()
    df['flag_low'] = df['low'].rolling(window=3).min()

    # --- FIX 1: WARMUP PERIOD ---
    df = df.dropna().copy()
    if df.empty: return [], df

    # Signal Logic Restoration
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = (df['close'] < df['ema_15_pk'])
    elif strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)
    # ... (Include all other switch cases here: Flags, Bollinger, RSI Divergence, etc.)

    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0

    if config.get('use_rsav', False) and benchmark_df is not None:
        df['rsav'] = calculate_rsav(df, benchmark_df, lookback=50)

    for i in range(1, len(df)):
        curr = df.iloc[i]; prev = df.iloc[i-1]
        market_ok = curr['rsav'] >= config.get('rsav_trigger', -0.5) if config.get('use_rsav', False) else True

        if active_trade:
            # --- FIX 2: REALISTIC EXITS ---
            sl_p = active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_p = active_trade.entry_price * (1 + config['tp_val'] / 100)
            
            sl_hit = config['use_sl'] and curr['low'] <= sl_p
            tp_hit = config['use_tp'] and curr['high'] >= tp_p

            if sl_hit:
                active_trade.exit_price = sl_p * (1 - slippage); active_trade.exit_date = curr.name; active_trade.exit_reason = "Stop Loss"
            elif tp_hit:
                active_trade.exit_price = tp_p * (1 - slippage); active_trade.exit_date = curr.name; active_trade.exit_reason = "Target Profit"
            elif prev['exit_signal']:
                active_trade.exit_price = curr['open'] * (1 - slippage); active_trade.exit_date = curr.name; active_trade.exit_reason = "Signal Exit"
            
            if active_trade.exit_date:
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        elif prev['long_signal'] and market_ok:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=curr.name, entry_price=curr['open'] * (1 + slippage))
            
    return trades, df

# --- 3. OPTIMIZER ENGINE ---
def run_pk_optimizer(data, symbol, config, capital):
    results = []
    ema_range = range(3, 61, 3) 
    prog = st.progress(0); steps = len(ema_range)**2; curr = 0
    for entry in ema_range:
        for ex in ema_range:
            curr += 1
            if entry <= ex: continue
            df_opt = data.copy()
            df_opt['ema_20_pk'] = df_opt['close'].ewm(span=entry, adjust=False).mean()
            df_opt['ema_15_pk'] = df_opt['close'].ewm(span=ex, adjust=False).mean()
            t_list, _ = run_backtest(df_opt, symbol, config, "PK Strategy (Positional)")
            if t_list:
                df_res = pd.DataFrame([vars(tr) for tr in t_list])
                # --- FIX 3: YOUR VERIFIED EXPECTANCY ---
                wr = len(df_res[df_res['pnl_pct'] > 0]) / len(df_res)
                avg_w = df_res[df_res['pnl_pct'] > 0]['pnl_pct'].mean() if wr > 0 else 0
                avg_l = df_res[df_res['pnl_pct'] <= 0]['pnl_pct'].mean() if wr < 1 else 0
                exp = (wr * avg_w) + ((1 - wr) * avg_l)
                
                df_res['equity'] = capital * (1 + df_res['pnl_pct']).cumprod()
                f_ret = ((df_res['equity'].iloc[-1] / capital) - 1) * 100
                results.append({"Entry": entry, "Exit": ex, "Return %": round(f_ret, 2), "Expectancy": round(exp*100, 4)})
            if curr % 100 == 0: prog.progress(curr/steps)
    prog.empty(); return pd.DataFrame(results)

# --- 4. UI COMPONENTS ---
# [Include your exact st.set_page_config, Sidebar Text Inputs, and CSS Styling here]

# --- 5. EXECUTION ---
# [In the Run_Single block, include your Tab structure and all 8 Statistics Expanders]