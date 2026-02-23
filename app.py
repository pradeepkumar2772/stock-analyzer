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

# --- 2. MULTI-STRATEGY ENGINE ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    # Pre-calculate Indicators
    df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # --- CANDLESTICK COMPONENT CALCULATIONS ---
    # We use .values here to prevent "identically-labeled" errors during shifts
    body_size = abs(df['close'] - df['open'])
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    
    # Nison Criterion: Lower shadow >= 2 * Body size
    is_umbrella = (lower_shadow >= 2 * body_size) & (upper_shadow <= body_size * 0.1)

    # --- FIX: WARMUP PROTECTION ---
    df = df.dropna().copy()
    if df.empty: return [], df

    # Re-align series for shifted comparisons
    c = df['close'].values
    o = df['open'].values
    l = df['low'].values
    h = df['high'].values
    
    # Signal Arrays
    is_hammer = is_umbrella.values & (c < pd.Series(c).shift(3).fillna(999999).values)
    
    # Bullish Engulfing
    prev_c = pd.Series(c).shift(1).values
    prev_o = pd.Series(o).shift(1).values
    is_bull_engulf = (c > o) & (prev_c < prev_o) & (o < prev_c) & (c > prev_o)
    
    # Piercing Pattern
    midpoint_prev = (prev_o + prev_c) / 2
    is_piercing = (c > o) & (prev_c < prev_o) & (o < pd.Series(l).shift(1).values) & (c > midpoint_prev)

    # Strategy Switch
    df['long_signal'] = False
    df['exit_signal'] = False

    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (pd.Series(c).shift(1).values < pd.Series(df['ema_20_pk']).shift(1).values) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = (df['close'] < df['ema_15_pk'])
    elif strategy_type == "Nison Ch. 4 (Classic Reversals)":
        df['long_signal'] = is_hammer | is_bull_engulf | is_piercing
        df['exit_signal'] = (df['close'] < pd.Series(l).shift(1).values)

    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        
        if active_trade:
            # REALISTIC EXIT LOGIC
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

# --- 3. UI STYLING ---
st.set_page_config(layout="wide", page_title="Pro-Tracer: Nison Edition")
st.markdown("<style>.stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; } .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; } .stat-label { color: #999; font-size: 0.85rem; } .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }</style>", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strategies_list = ["PK Strategy (Positional)", "Nison Ch. 4 (Classic Reversals)"]
strat_choice = st.sidebar.selectbox("Select Strategy", strategies_list)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2010-01-01")

config = {'slippage_val': 0.1, 'use_slippage': True, 'sl_val': 5.0, 'use_sl': True, 'tp_val': 25.0, 'use_tp': True}

# --- 5. EXECUTION ---
if st.sidebar.button("🚀 Run Backtest"):
    try:
        data = yf.download(symbol, start=start_str, auto_adjust=True)
        if not data.empty:
            # FIX: Ensure columns are lowercase before the engine starts
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            
            trades, _ = run_backtest(data.copy(), symbol, config, strat_choice)
            
            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                
                # VERIFIED THARP EXPECTANCY
                wins = df_trades[df_trades['pnl_pct'] > 0]
                losses = df_trades[df_trades['pnl_pct'] <= 0]
                wr = len(wins) / len(df_trades)
                avg_w = wins['pnl_pct'].mean() if not wins.empty else 0
                avg_l = losses['pnl_pct'].mean() if not losses.empty else 0
                exp = (wr * avg_w) + ((1 - wr) * avg_l)

                total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                
                st.subheader(f"Results for {symbol}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Return", f"{total_ret:.2f}%")
                c2.metric("Win Rate", f"{wr*100:.1f}%")
                c3.metric("Expectancy (Edge)", f"{exp:.4f}")
                
                st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Account Equity Curve"))
            else:
                st.warning("No patterns detected. Try a different symbol or longer date range.")
    except Exception as e:
        st.error(f"Execution Error: {e}")