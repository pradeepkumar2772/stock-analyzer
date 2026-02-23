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

# --- 2. MULTI-STRATEGY ENGINE ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    # Pre-calculate Indicators
    df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # --- CANDLESTICK COMPONENT CALCULATIONS [cite: 373-375] ---
    body_size = abs(df['close'] - df['open'])
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    midpoint_prev = (df['open'].shift(1) + df['close'].shift(1)) / 2
    
    # Umbrella Recognition: Small body, long lower shadow [cite: 507-509]
    is_umbrella = (lower_shadow >= 2 * body_size) & (upper_shadow <= body_size * 0.1)

    # --- FIX 1: WARMUP PROTECTION ---
    df = df.dropna().copy()
    if df.empty: return [], df

    # Initialize Signal Columns
    df['long_signal'] = False
    df['exit_signal'] = False

    # --- CHAPTER 4 STRATEGY LOGIC ---
    # Hammer: Umbrella in a downtrend [cite: 494, 527]
    is_hammer = is_umbrella & (df['close'] < df['close'].shift(3))
    
    # Bullish Engulfing: White body wraps black body [cite: 618-620, 625]
    is_bull_engulf = (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & \
                     (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))
    
    # Piercing Pattern: Opens below prior low, closes >50% into prior body [cite: 774-776, 781, 789]
    is_piercing = (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & \
                  (df['open'] < df['low'].shift(1)) & (df['close'] > midpoint_prev)

    # Strategy Switch Logic
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = (df['close'] < df['ema_15_pk'])
    elif strategy_type == "Nison Ch. 4 (Classic Reversals)":
        df['long_signal'] = is_hammer | is_bull_engulf | is_piercing
        df['exit_signal'] = (df['close'] < df['low'].shift(1)) # Simple low-break exit
    # ... (Other strategies remain identical)

    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        
        if active_trade:
            # --- FIX 2: REALISTIC EXIT FILL LOGIC ---
            sl_price = active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_price = active_trade.entry_price * (1 + config['tp_val'] / 100)
            
            sl_hit = config['use_sl'] and current['low'] <= sl_price
            tp_hit = config['use_tp'] and current['high'] >= tp_price

            if sl_hit:
                active_trade.exit_price, active_trade.exit_reason = sl_price * (1 - slippage), "Stop Loss"
            elif tp_hit:
                active_trade.exit_price, active_trade.exit_reason = tp_price * (1 - slippage), "Target Profit"
            elif prev['exit_signal']:
                active_trade.exit_price, active_trade.exit_reason = current['open'] * (1 - slippage), "Signal Exit"
            
            if active_trade.exit_price:
                active_trade.exit_date = current.name
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
            
    return trades, df

# --- 3. UI STYLING & SIDEBAR ---
st.set_page_config(layout="wide", page_title="Pro-Tracer: Nison Edition")
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strategies_list = ["PK Strategy (Positional)", "Nison Ch. 4 (Classic Reversals)", "RSI 60 Cross", "EMA Ribbon"]
strat_choice = st.sidebar.selectbox("Select Strategy", strategies_list)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2010-01-01")

config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 30, 'hhv_period': 20, 'slippage_val': 0.1, 'use_slippage': True, 'sl_val': 5.0, 'use_sl': True, 'tp_val': 25.0, 'use_tp': True}

# --- 5. EXECUTION & STATISTICS ---
if st.sidebar.button("🚀 Run Backtest"):
    try:
        # 1. Download data
        data = yf.download(symbol, start=start_str, auto_adjust=True)
        
        if not data.empty:
            # 2. STANDARDIZE COLUMNS BEFORE RUNNING BACKTEST
            # This fixes the KeyError: 'close'
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            
            # 3. Run the engine
            trades, processed_df = run_backtest(data.copy(), symbol, config, strat_choice)
            
            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                
                # --- THARP EXPECTANCY ---
                wins = df_trades[df_trades['pnl_pct'] > 0]
                losses = df_trades[df_trades['pnl_pct'] <= 0]
                wr = len(wins) / len(df_trades)
                avg_w = wins['pnl_pct'].mean() if not wins.empty else 0
                avg_l = losses['pnl_pct'].mean() if not losses.empty else 0
                exp = (wr * avg_w) + ((1 - wr) * avg_l)

                total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                
                # Display Results
                st.subheader(f"Results for {symbol}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Return", f"{total_ret:.2f}%")
                c2.metric("Win Rate", f"{wr*100:.1f}%")
                c3.metric("Expectancy", f"{exp:.4f}")
                
                st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Account Equity Curve"))
            else:
                st.warning("No trades found. The patterns from Nison Chapter 4 were not detected in this timeframe.")
        else:
            st.error("Could not fetch data. Please check the symbol or date range.")
    except Exception as e:
        st.error(f"Execution Error: {e}")