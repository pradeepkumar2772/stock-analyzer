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
    
    df['long_signal'] = False
    df['exit_signal'] = False
    
    # INDICATOR PROTECTION: Check if pre-calculated columns exist (for Optimizer)
    if 'ema_15_pk' not in df.columns:
        df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    if 'ema_20_pk' not in df.columns:
        df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()
        
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()
    # (Rest of indicator calculations remain same as your base...)
    df['sma_200'] = df['close'].rolling(window=200).mean()

    # PK Strategy Logic
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = (df['close'] < df['ema_15_pk'])
    # (Additional strategies from your base code...)

    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            if sl_hit or tp_hit or prev['exit_signal']:
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
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
            t, _ = run_backtest(df_opt, symbol, config, "PK Strategy (Positional)")
            if t:
                df_res = pd.DataFrame([vars(tr) for tr in t])
                df_res['equity'] = capital * (1 + df_res['pnl_pct']).cumprod()
                f_ret = ((df_res['equity'].iloc[-1] / capital) - 1) * 100
                mdd = ((df_res['equity'] - df_res['equity'].cummax()) / df_res['equity'].cummax()).min() * 100
                results.append({"Entry": entry, "Exit": ex, "Return %": round(f_ret, 2), "Max DD %": round(mdd, 2), "RF": round(f_ret / abs(mdd) if mdd != 0 else f_ret, 2)})
            if curr % 100 == 0: prog.progress(curr/steps)
    prog.empty(); return pd.DataFrame(results)

# --- 4. SIDEBAR & BUTTONS ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strat_choice = st.sidebar.selectbox("Select Strategy", ["PK Strategy (Positional)", "RSI 60 Cross", "EMA Ribbon"])
selected_tf = st.sidebar.selectbox("Timeframe", ["Daily", "Weekly", "1 Hour"], index=0)
tf_map = {"Daily": "1d", "Weekly": "1wk", "1 Hour": "1h"}
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2010-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

config = {'sl_val': 5.0, 'use_sl': True, 'tp_val': 25.0, 'use_tp': True, 'use_slippage': True, 'slippage_val': 0.1, 'use_rsav': False}

col1, col2, col3 = st.sidebar.columns(3)
run_single = col1.button("🚀 Backtest")
run_arena = col2.button("🏟️ Arena")
run_opt = col3.button("🎯 Optimizer")

# --- 5. EXECUTION BLOCK ---
if run_single or run_arena or run_opt:
    data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        
        # UNIVERSAL BASELINE (Sync Point)
        baseline_df = data.copy()
        baseline_df['ema_20_pk'] = baseline_df['close'].ewm(span=20, adjust=False).mean()
        baseline_df['ema_15_pk'] = baseline_df['close'].ewm(span=15, adjust=False).mean()

        if run_single:
            trades, _ = run_backtest(baseline_df, symbol, config, strat_choice)
            if trades:
                df = pd.DataFrame([vars(t) for t in trades])
                df['equity'] = capital * (1 + df['pnl_pct']).cumprod()
                st.metric("Total Return", f"{(((df['equity'].iloc[-1]/capital)-1)*100):.2f}%")
                st.plotly_chart(px.line(df, x='exit_date', y='equity', title="Equity Curve"), use_container_width=True)

        elif run_opt:
            b_trades, _ = run_backtest(baseline_df.copy(), symbol, config, "PK Strategy (Positional)")
            b_ret = (((pd.DataFrame([vars(t) for t in b_trades])['pnl_pct'].add(1).prod()) - 1) * 100) if b_trades else 0
            
            opt_df = run_pk_optimizer(data, symbol, config, capital)
            if not opt_df.empty:
                best = opt_df.loc[opt_df['RF'].idxmax()]
                st.subheader(f"🎯 Optimization Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("Baseline (20/15)", f"{b_ret:.2f}%")
                c2.metric("Best Optimized", f"{best['Return %']:.2f}%", delta=f"{best['Return %'] - b_ret:.2f}%")
                c3.metric("Best Combo", f"{best['Entry']} / {best['Exit']}")
                
                st.dataframe(
                    opt_df.sort_values(by="RF", ascending=False).head(10),
                    column_config={"RF": st.column_config.ProgressColumn(min_value=0, max_value=float(opt_df['RF'].max()))},
                    use_container_width=True, hide_index=True
                )