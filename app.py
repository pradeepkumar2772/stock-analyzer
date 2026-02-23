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

# --- 2. ENGINES (Shared by both pages) ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    df = df.copy()
    # Indicators
    df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    # ... (Add other indicators from your stable code here)
    
    # FIX: Warmup Period
    df = df.dropna().copy()
    if df.empty: return [], df

    # Signal Logic (PK Strategy Example)
    df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
    df['exit_signal'] = (df['close'] < df['ema_15_pk'])

    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0

    for i in range(1, len(df)):
        curr = df.iloc[i]; prev = df.iloc[i-1]
        if active_trade:
            # FIX: Realistic Exit Prices
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
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=curr.name, entry_price=curr['open'] * (1 + slippage))
    return trades, df

# --- 3. PAGE CONFIG & NAVIGATION ---
st.set_page_config(layout="wide", page_title="Pro-Tracer V4")
page = st.sidebar.radio("Navigation", ["🚀 Run Backtest", "🏟️ Strategy Arena"])

# Common Sidebar Inputs
st.sidebar.divider()
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_date = st.sidebar.date_input("Start Date", date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())
config = {'sl_val': 5.0, 'use_sl': True, 'tp_val': 25.0, 'use_tp': True, 'use_slippage': True, 'slippage_val': 0.1}

# --- 4. PAGE 1: RUN BACKTEST ---
if page == "🚀 Run Backtest":
    st.title("Single Strategy Analysis")
    strat = st.selectbox("Select Strategy", ["PK Strategy (Positional)", "RSI 60 Cross"])
    
    if st.button("Calculate Performance"):
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
        if not data.empty:
            data.columns = [str(col).lower() for col in data.columns]
            trades, _ = run_backtest(data, symbol, config, strat)
            if trades:
                df_t = pd.DataFrame([vars(t) for t in trades])
                df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
                
                # Tharp Expectancy Fix
                wr = len(df_t[df_t['pnl_pct'] > 0]) / len(df_t)
                exp = (wr * df_t[df_t['pnl_pct'] > 0]['pnl_pct'].mean()) + ((1 - wr) * df_t[df_t['pnl_pct'] <= 0]['pnl_pct'].mean())
                
                st.metric("Expectancy (Edge)", f"{exp*100:.2f}%")
                st.plotly_chart(px.line(df_t, x='exit_date', y='equity'))
            else: st.warning("No trades found.")

# --- 5. PAGE 2: STRATEGY ARENA ---
elif page == "🏟️ Strategy Arena":
    st.title("The Strategy Arena")
    st.caption("Comparing all strategies side-by-side")
    
    if st.button("Start Tournament"):
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
        if not data.empty:
            data.columns = [str(col).lower() for col in data.columns]
            arena_results = []
            
            for s_name in ["PK Strategy (Positional)", "RSI 60 Cross"]:
                t, _ = run_backtest(data.copy(), symbol, config, s_name)
                if t:
                    res_df = pd.DataFrame([vars(tr) for tr in t])
                    ret = (res_df['pnl_pct'].add(1).prod() - 1) * 100
                    arena_results.append({"Strategy": s_name, "Return %": round(ret, 2)})
            
            st.table(pd.DataFrame(arena_results).sort_values("Return %", ascending=False))