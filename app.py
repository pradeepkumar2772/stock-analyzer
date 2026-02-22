import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date, timedelta

# --- 1. CORE DATA STRUCTURE (Identical) ---
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

# --- RS-AV CALCULATION ENGINE (Identical) ---
def calculate_rsav(stock_df, benchmark_df, lookback=50):
    s_roc = stock_df['close'].pct_change(lookback) * 100
    s_vol = s_roc.rolling(window=lookback).std()
    s_net = s_roc - s_vol
    b_roc = benchmark_df['close'].pct_change(lookback) * 100
    b_vol = b_roc.rolling(window=lookback).std()
    b_net = b_roc - b_vol
    return s_net - b_net

# --- 2. MULTI-STRATEGY ENGINE (Identical) ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    df['long_signal'] = False
    df['exit_signal'] = False
    
    if config.get('use_rsav', False) and benchmark_df is not None:
        df['rsav'] = calculate_rsav(df, benchmark_df, lookback=50)

    # Indicator Protection for Optimizer: Only calculate if not already present
    if 'ema_15_pk' not in df.columns:
        df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    if 'ema_20_pk' not in df.columns:
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

    # Strategy Switch Logic
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = (df['close'] < df['ema_15_pk'])
    elif strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)
    elif strategy_type == "EMA Ribbon":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = (df['ema_fast'] < df['ema_exit'])

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

# --- 3. UI STYLING ---
st.set_page_config(layout="wide", page_title="Strategy Lab Pro")
def draw_stat(label, value):
    st.markdown(f"<div style='display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #2d2f3b;'><span style='color:#999;'>{label}</span><span style='color:#fff; font-weight:600;'>{value}</span></div>", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strategies_list = ["PK Strategy (Positional)", "RSI 60 Cross", "EMA Ribbon"]
strat_choice = st.sidebar.selectbox("Select Strategy", strategies_list)
tf_map = {"1 Minute": "1m", "5 Minutes": "5m", "Daily": "1d"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=2)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2020-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

use_rsav = st.sidebar.toggle("Enable RS-AV Filter", False)
config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 30, 'use_rsav': use_rsav, 'rsav_trigger': -0.5, 'sl_val': 5.0, 'tp_val': 25.0, 'use_sl': True, 'use_tp': True, 'use_slippage': True, 'slippage_val': 0.1}

col_run1, col_run2, col_run3 = st.sidebar.columns(3)
run_single = col_run1.button("🚀 Run Backtest")
run_arena = col_run2.button("🏟️ Run Arena")
run_opt = col_run3.button("🎯 Run Optimizer")

# --- 5. EXECUTION LOGIC ---
if run_single:
    data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        trades, _ = run_backtest(data.copy(), symbol, config, strat_choice)
        if trades:
            df_trades = pd.DataFrame([vars(t) for t in trades])
            df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
            peak = df_trades['equity'].cummax()
            mdd = ((df_trades['equity'] - peak) / peak).min() * 100
            st.metric("Total Return", f"{((df_trades['equity'].iloc[-1]/capital-1)*100):.2f}%", delta=f"{mdd:.2f}% Max DD")
            st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title=f"Equity Curve: {strat_choice}"), use_container_width=True)
        else: st.warning("No trades found.")

elif run_arena:
    data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        arena_results = []
        for s in strategies_list:
            t, _ = run_backtest(data.copy(), symbol, config, s)
            if t:
                df_temp = pd.DataFrame([vars(tr) for tr in t])
                df_temp['equity'] = capital * (1 + df_temp['pnl_pct']).cumprod()
                ret = ((df_temp['equity'].iloc[-1] / capital) - 1) * 100
            else: ret = 0
            arena_results.append({"Strategy": s, "Return %": round(ret, 2)})
        st.table(pd.DataFrame(arena_results).sort_values(by="Return %", ascending=False))

elif run_opt:
    data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        
        # Baseline Comparison (Matches Single Backtest)
        b_trades, _ = run_backtest(data.copy(), symbol, config, "PK Strategy (Positional)")
        b_ret = 0.0
        if b_trades:
            b_df = pd.DataFrame([vars(t) for t in b_trades])
            b_df['equity'] = capital * (1 + b_df['pnl_pct']).cumprod()
            b_ret = ((b_df['equity'].iloc[-1] / capital) - 1) * 100

        # Optimization Loop
        opt_results = []
        ema_range = range(3, 61, 3)
        prog = st.progress(0)
        steps = len(ema_range)**2; curr = 0

        for entry in ema_range:
            for ex in ema_range:
                curr += 1
                if entry <= ex: continue
                df_opt = data.copy()
                df_opt['ema_20_pk'] = df_opt['close'].ewm(span=entry, adjust=False).mean()
                df_opt['ema_15_pk'] = df_opt['close'].ewm(span=ex, adjust=False).mean()
                t_opt, _ = run_backtest(df_opt, symbol, config, "PK Strategy (Positional)")
                if t_opt:
                    res = pd.DataFrame([vars(tr) for tr in t_opt])
                    res['equity'] = capital * (1 + res['pnl_pct']).cumprod()
                    f_ret = ((res['equity'].iloc[-1] / capital) - 1) * 100
                    pk = res['equity'].cummax()
                    mdd = ((res['equity'] - pk) / pk).min() * 100
                    opt_results.append({"Entry": entry, "Exit": ex, "Return %": round(f_ret, 2), "Max DD %": round(mdd, 2), "RF": round(f_ret / abs(mdd) if mdd != 0 else f_ret, 2)})
                if curr % 50 == 0: prog.progress(curr/steps)
        
        prog.empty()
        res_df = pd.DataFrame(opt_results)
        if not res_df.empty:
            best = res_df.loc[res_df['RF'].idxmax()]
            st.subheader(f"🎯 Optimization Summary: {symbol}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Original Baseline", f"{b_ret:.2f}%")
            c2.metric("Best Optimized", f"{best['Return %']:.2f}%", delta=f"{best['Return %'] - b_ret:.2f}%")
            c3.metric("Best Combo", f"{best['Entry']} / {best['Exit']}")
            st.dataframe(res_df.sort_values(by="RF", ascending=False).head(10), column_config={"RF": st.column_config.ProgressColumn(min_value=0, max_value=float(res_df['RF'].max()))}, use_container_width=True, hide_index=True)