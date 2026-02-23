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

# --- 2. MULTI-STRATEGY ENGINE (Fixed & Robust) ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    df = df.copy()
    
    # Indicator Logic Protection
    if 'ema_15_pk' not in df.columns:
        df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    if 'ema_20_pk' not in df.columns:
        df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Restore standard indicators
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # --- FIX: Warmup Period ---
    df = df.dropna().copy()
    if df.empty: return [], df

    # Signal logic for PK Strategy
    df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
    df['exit_signal'] = (df['close'] < df['ema_15_pk'])

    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        if active_trade:
            # --- FIX: Realistic Exit Fills ---
            sl_p = active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_p = active_trade.entry_price * (1 + config['tp_val'] / 100)
            
            sl_hit = config['use_sl'] and curr['low'] <= sl_p
            tp_hit = config['use_tp'] and curr['high'] >= tp_p
            
            if sl_hit:
                active_trade.exit_price = sl_p * (1 - slippage)
                active_trade.exit_date = curr.name
            elif tp_hit:
                active_trade.exit_price = tp_p * (1 - slippage)
                active_trade.exit_date = curr.name
            elif prev['exit_signal']:
                active_trade.exit_price = curr['open'] * (1 - slippage)
                active_trade.exit_date = curr.name
                
            if active_trade.exit_date:
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None
        elif prev['long_signal']:
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
            
            # --- FIX: Error Protection for Zero Trades ---
            if t_list:
                df_res = pd.DataFrame([vars(tr) for tr in t_list])
                count = len(df_res)
                
                # Verified Expectancy Formula
                wins = df_res[df_res['pnl_pct'] > 0]
                losses = df_res[df_res['pnl_pct'] <= 0]
                wr = len(wins) / count
                avg_w = wins['pnl_pct'].mean() if not wins.empty else 0
                avg_l = losses['pnl_pct'].mean() if not losses.empty else 0
                exp = (wr * avg_w) + ((1 - wr) * avg_l)
                
                df_res['equity'] = capital * (1 + df_res['pnl_pct']).cumprod()
                f_ret = ((df_res['equity'].iloc[-1] / capital) - 1) * 100
                results.append({"Entry": entry, "Exit": ex, "Return %": round(f_ret, 2), "Expectancy": round(exp*100, 4)})
            
            if curr % 100 == 0: prog.progress(curr/steps)
    prog.empty(); return pd.DataFrame(results)

# --- 4. SIDEBAR & EXECUTION ---
st.set_page_config(layout="wide", page_title="Pro-Tracer")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2010-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))
config = {'sl_val': 5.0, 'use_sl': True, 'tp_val': 25.0, 'use_tp': True, 'use_slippage': True, 'slippage_val': 0.1}

col1, col2, col3 = st.sidebar.columns(3)
run_single = col1.button("🚀 Backtest")
run_opt = col3.button("🎯 Optimizer")

if run_single or run_opt:
    data = yf.download(symbol, start=start_str, end=end_str, interval="1d", auto_adjust=True)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        
        if run_single:
            trades, _ = run_backtest(data.copy(), symbol, config, "PK Strategy (Positional)")
            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Equity Curve"), use_container_width=True)
            else: st.warning("No trades found for this symbol.")

        elif run_opt:
            opt_df = run_pk_optimizer(data, symbol, config, capital)
            if not opt_df.empty:
                st.subheader("🎯 Optimizer Results")
                st.dataframe(opt_df.sort_values("Expectancy", ascending=False).head(20), use_container_width=True)