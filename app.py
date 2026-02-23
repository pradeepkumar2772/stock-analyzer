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

# --- 2. MULTI-STRATEGY ENGINE (Fixed Discrepancies) ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    df = df.copy()
    
    # Logic Protection: Check if pre-calculated columns exist (for Optimizer injection)
    if 'ema_15_pk' not in df.columns:
        df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    if 'ema_20_pk' not in df.columns:
        df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()

    # Restoration of Indicators
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

    # --- FIX 1: WARMUP PERIOD PROTECTION ---
    # Removes initial rows where indicators (like SMA 200) are NaN
    df = df.dropna().copy()
    if df.empty: return [], df

    # Signal Initialization
    df['long_signal'] = False
    df['exit_signal'] = False
    
    if config.get('use_rsav', False) and benchmark_df is not None:
        df['rsav'] = calculate_rsav(df, benchmark_df, lookback=50)

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
    elif strategy_type == "Relative Strength Play":
        stock_ret = df['close'].pct_change(periods=55)
        df['long_signal'] = (stock_ret > 0) & (df['close'] > df['ema_fast'])
        df['exit_signal'] = (df['close'] < df['ema_slow'])

    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        if active_trade:
            # --- FIX 2: REALISTIC EXIT LOGIC (Fills at trigger price, not Open) ---
            sl_price = active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_price = active_trade.entry_price * (1 + config['tp_val'] / 100)
            
            sl_hit = config['use_sl'] and curr['low'] <= sl_price
            tp_hit = config['use_tp'] and curr['high'] >= tp_price
            
            if sl_hit:
                active_trade.exit_price = sl_price * (1 - slippage)
                active_trade.exit_date = curr.name
                active_trade.exit_reason = "Stop Loss"
            elif tp_hit:
                active_trade.exit_price = tp_price * (1 - slippage)
                active_trade.exit_date = curr.name
                active_trade.exit_reason = "Target Profit"
            elif prev['exit_signal']:
                active_trade.exit_price = curr['open'] * (1 - slippage)
                active_trade.exit_date = curr.name
                active_trade.exit_reason = "Signal Exit"
                
            if active_trade.exit_date:
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None
        
        elif prev['long_signal']:
            # RS-AV filter check
            market_ok = curr['rsav'] >= config.get('rsav_trigger', -0.5) if config.get('use_rsav', False) else True
            if market_ok:
                active_trade = Trade(
                    symbol=symbol, direction="Long", 
                    entry_date=curr.name, 
                    entry_price=curr['open'] * (1 + slippage)
                )
            
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
                # --- FIX 3: YOUR VERIFIED EXPECTANCY FORMULA ---
                wr = len(df_res[df_res['pnl_pct'] > 0]) / len(df_res)
                avg_w = df_res[df_res['pnl_pct'] > 0]['pnl_pct'].mean() if wr > 0 else 0
                avg_l = df_res[df_res['pnl_pct'] <= 0]['pnl_pct'].mean() if wr < 1 else 0
                exp = (wr * avg_w) + ((1 - wr) * avg_l)
                
                df_res['equity'] = capital * (1 + df_res['pnl_pct']).cumprod()
                f_ret = ((df_res['equity'].iloc[-1] / capital) - 1) * 100
                results.append({"Entry": entry, "Exit": ex, "Return %": round(f_ret, 2), "Expectancy": round(exp*100, 3)})
            if curr % 100 == 0: prog.progress(curr/steps)
    prog.empty(); return pd.DataFrame(results)

# --- 4. UI STYLING ---
st.set_page_config(layout="wide", page_title="Strategy Lab Pro")
st.markdown("<style>.stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; } .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; } .stat-label { color: #999; font-size: 0.85rem; } .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }</style>", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- 5. SIDEBAR ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strategies_list = ["PK Strategy (Positional)", "RSI 60 Cross", "EMA Ribbon", "Relative Strength Play"]
strat_choice = st.sidebar.selectbox("Select Strategy", strategies_list)
tf_map = {"Daily": "1d", "Weekly": "1wk"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=0)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2010-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

st.sidebar.divider()
use_rsav = st.sidebar.toggle("Enable RS-AV Filter", False)
rsav_trig = st.sidebar.number_input("RS-AV Trigger", value=-0.5, step=0.1)

config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 30, 'hhv_period': 20, 'use_rsav': use_rsav, 'rsav_trigger': rsav_trig}
st.sidebar.divider()
use_sl = st.sidebar.toggle("Stop Loss", True); config['sl_val'] = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0; config['use_sl'] = use_sl
use_tp = st.sidebar.toggle("Target Profit", True); config['tp_val'] = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0; config['use_tp'] = use_tp
use_slip = st.sidebar.toggle("Slippage", True); config['slippage_val'] = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slip else 0; config['use_slippage'] = use_slip

# --- 6. EXECUTION ---
col_run1, col_run2, col_run3 = st.sidebar.columns(3)
run_single = col_run1.button("🚀 Backtest")
run_arena = col_run2.button("🏟️ Arena")
run_opt = col_run3.button("🎯 Optimizer")

if run_single:
    data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        
        trades, _ = run_backtest(data.copy(), symbol, config, strat_choice)
        if trades:
            df_trades = pd.DataFrame([vars(t) for t in trades])
            df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
            
            # Metric Calculation
            wr = len(df_trades[df_trades['pnl_pct'] > 0]) / len(df_trades)
            exp_val = (wr * df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean()) + ((1 - wr) * df_trades[df_trades['pnl_pct'] <= 0]['pnl_pct'].mean())
            total_ret = ((df_trades['equity'].iloc[-1]/capital)-1)*100
            peak = df_trades['equity'].cummax(); mdd = ((df_trades['equity'] - peak) / peak).min() * 100
            
            st.metric("Total Return", f"{total_ret:.2f}%", delta=f"{mdd:.2f}% MDD")
            st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Strategy Equity Curve"), use_container_width=True)

            with st.expander("📊 8 Professional Expanders", expanded=True):
                cl, cr = st.columns(2)
                with cl:
                    draw_stat("Win Rate", f"{wr*100:.1f}%")
                    draw_stat("Expectancy (Edge)", f"{exp_val*100:.2f}%")
                    draw_stat("Max Drawdown", f"{mdd:.2f}%")
                with cr:
                    draw_stat("Total Trades", len(df_trades))
                    draw_stat("Calmar Ratio", f"{abs(total_ret/mdd):.2f}")

elif run_opt:
    data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        
        opt_df = run_pk_optimizer(data, symbol, config, capital)
        if not opt_df.empty:
            st.subheader("🎯 Optimizer Results (Ranked by Expectancy)")
            st.dataframe(opt_df.sort_values("Expectancy", ascending=False).head(20), 
                         column_config={"Expectancy": st.column_config.ProgressColumn(min_value=float(opt_df['Expectancy'].min()), max_value=float(opt_df['Expectancy'].max()))},
                         use_container_width=True, hide_index=True)