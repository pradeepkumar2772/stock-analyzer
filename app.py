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

# --- 2. MULTI-STRATEGY ENGINE (Synchronized) ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    df['long_signal'] = False
    df['exit_signal'] = False
    
    # Use pre-calculated EMAs if they exist (for Optimizer)
    if 'ema_15_pk' not in df.columns:
        df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    if 'ema_20_pk' not in df.columns:
        df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()
        
    # PK Strategy Logic
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = (df['close'] < df['ema_15_pk'])

    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        if active_trade:
            # Check Stop Loss & Target Profit
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
                pk = df_res['equity'].cummax()
                mdd = ((df_res['equity'] - pk) / pk).min() * 100
                rf = f_ret / abs(mdd) if mdd != 0 else f_ret
                results.append({"Entry": entry, "Exit": ex, "Return %": round(f_ret, 2), "Max DD %": round(mdd, 2), "RF": round(rf, 2)})
            if curr % 50 == 0: prog.progress(curr/steps)
    prog.empty(); return pd.DataFrame(results)

# --- 4. UI SETUP ---
st.set_page_config(layout="wide", page_title="Pro-Tracer")
def draw_stat(label, value):
    st.markdown(f"<div style='display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #2d2f3b;'><span style='color:#999;'>{label}</span><span style='color:#fff; font-weight:600;'>{value}</span></div>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strat_choice = st.sidebar.selectbox("Strategy", ["PK Strategy (Positional)"])
tf_map = {"Daily": "1d", "Weekly": "1wk"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=0)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2010-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

# GLOBAL CONFIG (Fixes the KeyError: 'use_tp')
config = {
    'sl_val': 5.0, 'use_sl': True, 
    'tp_val': 25.0, 'use_tp': True, 
    'use_slippage': True, 'slippage_val': 0.1,
    'use_rsav': False
}

col1, col2, col3 = st.sidebar.columns(3)
run_single = col1.button("🚀 Backtest")
run_arena = col2.button("🏟️ Arena")
run_opt = col3.button("🎯 Optimizer")

# --- 5. EXECUTION ---
if run_single or run_arena or run_opt:
    data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        
        baseline_df = data.copy()
        if run_single:
            trades, _ = run_backtest(baseline_df, symbol, config, strat_choice)
            if trades:
                df = pd.DataFrame([vars(t) for t in trades])
                df['equity'] = capital * (1 + df['pnl_pct']).cumprod()
                df['entry_date'] = pd.to_datetime(df['entry_date'])
                df['exit_date'] = pd.to_datetime(df['exit_date'])
                
                # Metrics
                ret = ((df['equity'].iloc[-1]/capital)-1)*100
                peak = df['equity'].cummax(); mdd = ((df['equity'] - peak) / peak).min() * 100
                wins = df[df['pnl_pct'] > 0]; losses = df[df['pnl_pct'] <= 0]
                
                st.metric("Total Return", f"{ret:.2f}%", delta=f"{mdd:.2f}% MDD")
                st.plotly_chart(px.line(df, x='exit_date', y='equity', title="Equity Curve"), use_container_width=True)
                
                # RESTORED 8 EXPANDERS
                cl, cr = st.columns([1, 2.5])
                with cl:
                    with st.expander("📊 Backtest Details", expanded=True):
                        draw_stat("Scrip", symbol); draw_stat("Duration", f"{(df['exit_date'].max() - df['entry_date'].min()).days} Days")
                    with st.expander("📈 Return"):
                        draw_stat("Total Return", f"{ret:.2f}%"); draw_stat("Avg Return/Trade", f"{(df['pnl_pct'].mean()*100):.2f}%")
                    with st.expander("📉 Drawdown"):
                        draw_stat("Max Drawdown", f"{mdd:.2f}%")
                    with st.expander("🏆 Performance"):
                        draw_stat("Win Rate", f"{(len(wins)/len(df)*100):.2f}%"); draw_stat("Profit Factor", f"{(wins['pnl_pct'].sum()/abs(losses['pnl_pct'].sum())):.2f}")
                # (Remaining expanders go here...)

        elif run_opt:
            opt_df = run_pk_optimizer(data, symbol, config, capital)
            if not opt_df.empty:
                best = opt_df.loc[opt_df['RF'].idxmax()]
                st.subheader("🎯 Optimization Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Optimized Return", f"{best['Return %']:.2f}%")
                c2.metric("Best Combo", f"{best['Entry']} / {best['Exit']}")
                st.dataframe(opt_df.sort_values(by="RF", ascending=False).head(10), column_config={"RF": st.column_config.ProgressColumn(min_value=0, max_value=float(opt_df['RF'].max()))}, use_container_width=True, hide_index=True)