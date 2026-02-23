import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date

# -----------------------------
# 1. TRADE DATA STRUCTURE
# -----------------------------
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


# -----------------------------
# 2. RS-AV CALCULATION
# -----------------------------
def calculate_rsav(stock_df, benchmark_df, lookback=50):
    s_roc = stock_df['close'].pct_change(lookback) * 100
    s_vol = s_roc.rolling(window=lookback).std()
    s_net = s_roc - s_vol

    b_roc = benchmark_df['close'].pct_change(lookback) * 100
    b_vol = b_roc.rolling(window=lookback).std()
    b_net = b_roc - b_vol

    return s_net - b_net


# -----------------------------
# 3. BACKTEST ENGINE (FIXED)
# -----------------------------
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):

    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config.get('use_slippage', False) else 0

    df = df.copy()
    df['long_signal'] = False
    df['exit_signal'] = False

    # --- RS-AV ---
    if config.get('use_rsav', False) and benchmark_df is not None:
        df['rsav'] = calculate_rsav(df, benchmark_df, lookback=50)

    # --- Indicators ---
    df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()

    df['sma_200'] = df['close'].rolling(200).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['std_20'] = df['close'].rolling(20).std()
    df['upper_bb'] = df['sma_20'] + (df['std_20'] * 2)
    df['lower_bb'] = df['sma_20'] - (df['std_20'] * 2)
    df['bb_width'] = df['upper_bb'] - df['lower_bb']

    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    df['hhv'] = df['high'].rolling(config.get('hhv_period', 20)).max()
    df['llv'] = df['low'].rolling(config.get('hhv_period', 20)).min()
    df['neckline'] = df['high'].rolling(20).max()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = abs(delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    df['pole_return'] = df['close'].pct_change(10)
    df['is_pole'] = df['pole_return'] > 0.08
    df['flag_high'] = df['high'].rolling(3).max()
    df['flag_low'] = df['low'].rolling(3).min()

    # 🔥 Remove warm-up NaNs
    df = df.dropna().copy()

    # ---------------- STRATEGY SWITCH ----------------
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = df['close'] < df['ema_15_pk']

    elif strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)

    elif strategy_type == "EMA Ribbon":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = df['ema_fast'] < df['ema_exit']

    # --------------------------------------------------

    # ---------------- EXECUTION LOOP ----------------
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i - 1]

        market_ok = True
        if config.get('use_rsav', False) and 'rsav' in df.columns:
            market_ok = current['rsav'] >= config.get('rsav_trigger', -0.5)

        if active_trade:

            sl_price = active_trade.entry_price * (1 - config.get('sl_val', 0) / 100)
            tp_price = active_trade.entry_price * (1 + config.get('tp_val', 0) / 100)

            sl_hit = config.get('use_sl', False) and current['low'] <= sl_price
            tp_hit = config.get('use_tp', False) and current['high'] >= tp_price

            if sl_hit or tp_hit or prev['exit_signal']:

                if sl_hit:
                    exit_price = sl_price
                    reason = "Stop Loss"

                elif tp_hit:
                    exit_price = tp_price
                    reason = "Target Profit"

                else:
                    exit_price = current['open']
                    reason = "Signal Exit"

                exit_price *= (1 - slippage)

                active_trade.exit_price = exit_price
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (
                    (exit_price - active_trade.entry_price)
                    / active_trade.entry_price
                )

                trades.append(active_trade)
                active_trade = None

        elif prev['long_signal'] and market_ok:

            entry_price = current['open'] * (1 + slippage)

            active_trade = Trade(
                symbol=symbol,
                direction="Long",
                entry_date=current.name,
                entry_price=entry_price
            )

    return trades, df