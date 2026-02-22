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
    pnl_pct: float = 0.0

# --- 2. MULTI-STRATEGY ENGINE (Locked for Consistency) ---
def run_backtest(df, symbol, config, strategy_type):
    trades = []
    active_trade = None
    # Ensure slippage is a decimal (0.1% -> 0.001)
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    # Use pre-calculated columns if they exist (to allow Optimizer to override)
    if 'ema_20_pk' not in df.columns:
        df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()
    if 'ema_15_pk' not in df.columns:
        df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()

    # Signals
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = (df['close'] < df['ema_15_pk'])

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            if sl_hit or prev['exit_signal']:
                # Applying slippage to the exit price
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None
        elif prev['long_signal']:
            # Applying slippage to the entry price
            active_trade = Trade(
                symbol=symbol, 
                direction="Long", 
                entry_date=current.name, 
                entry_price=current['open'] * (1 + slippage)
            )
    return trades

# --- 3. THE OPTIMIZER MODULE ---
def run_pk_optimizer(data, symbol, config, capital):
    results = []
    ema_range = range(3, 61, 3) 
    prog = st.progress(0)
    steps = len(ema_range)**2
    curr = 0
    
    for entry in ema_range:
        for ex in ema_range:
            curr += 1
            if entry <= ex: continue
            
            df_opt = data.copy()
            df_opt['ema_20_pk'] = df_opt['close'].ewm(span=entry, adjust=False).mean()
            df_opt['ema_15_pk'] = df_opt['close'].ewm(span=ex, adjust=False).mean()
            
            t = run_backtest(df_opt, symbol, config, "PK Strategy (Positional)")
            
            if t:
                df_res = pd.DataFrame([vars(tr) for tr in t])
                # COMPOUNDING FORMULA (Matches Backtester exactly)
                df_res['equity'] = capital * (1 + df_res['pnl_pct']).cumprod()
                f_ret = ((df_res['equity'].iloc[-1] / capital) - 1) * 100
                pk = df_res['equity'].cummax()
                mdd = ((df_res['equity'] - pk) / pk).min() * 100
                rf = f_ret / abs(mdd) if mdd != 0 else f_ret
                
                results.append({"Entry": entry, "Exit": ex, "Return %": round(f_ret, 2), "Max DD %": round(mdd, 2), "RF": round(rf, 2)})
            if curr % 50 == 0: prog.progress(curr/steps)
    prog.empty()
    return pd.DataFrame(results)

# --- 4. EXECUTION ---
# (Assumes standard sidebar inputs for symbol_input, start_str, end_str, selected_tf, capital)
if run_single or run_arena or run_opt:
    data = yf.download(symbol_input, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
    if not data.empty:
        # Clean columns
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        
        # SHARED CALCULATION (Sync Point)
        baseline_df = data.copy()
        baseline_df['ema_20_pk'] = baseline_df['close'].ewm(span=20, adjust=False).mean()
        baseline_df['ema_15_pk'] = baseline_df['close'].ewm(span=15, adjust=False).mean()

        if run_single:
            t = run_backtest(baseline_df, symbol_input, config, strat_choice)
            if t:
                df = pd.DataFrame([vars(tr) for tr in t])
                df['equity'] = capital * (1 + df['pnl_pct']).cumprod()
                ret = ((df['equity'].iloc[-1]/capital)-1)*100
                st.metric("Unified Return", f"{ret:.2f}%")
                st.plotly_chart(px.line(df, x='exit_date', y='equity', title="Matched Equity Curve"), use_container_width=True)

        elif run_opt:
            # Baseline Check
            t_base = run_backtest(baseline_df.copy(), symbol_input, config, "PK Strategy (Positional)")
            b_ret = 0.0
            if t_base:
                df_b = pd.DataFrame([vars(tr) for tr in t_base])
                df_b['equity'] = capital * (1 + df_b['pnl_pct']).cumprod()
                b_ret = ((df_b['equity'].iloc[-1] / capital) - 1) * 100

            opt_df = run_pk_optimizer(data, symbol_input, config, capital)
            if not opt_df.empty:
                best = opt_df.loc[opt_df['RF'].idxmax()]
                st.metric("Baseline Check", f"{b_ret:.2f}% (Should match single backtest)")
                st.metric("Best Optimized", f"{best['Return %']:.2f}%", delta=f"{best['Return %'] - b_ret:.2f}%")
                st.dataframe(opt_df.sort_values(by="RF", ascending=False).head(10), hide_index=True)