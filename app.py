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
def run_backtest(df, symbol, config, strategy_type):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    # --- Indicator Calculations ---
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    # PK Strategy EMAs
    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    # Chandelier Exit (ATR Trailing Stop) Logic
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    highest_high = 0

    # HHV / LLV / Neckline
    h_per = config.get('hhv_period', 20)
    df['hhv'] = df['high'].rolling(window=h_per).max()
    df['llv'] = df['low'].rolling(window=h_per).min()
    df['neckline'] = df['high'].rolling(window=20).max()

    # --- Strategy Signals ---
    if strategy_type == "PK Strategy (Positional)":
        # Entry: Prev Close < EMA20 and Current Close > EMA20
        df['long_signal'] = (df['close'].shift(1) < df['ema_20'].shift(1)) & (df['close'] > df['ema_20'])
        # Exit: Close < EMA15
        df['exit_signal'] = (df['close'] < df['ema_15'])

    elif strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)

    elif strategy_type == "EMA Ribbon":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = (df['ema_fast'] < df['ema_exit'])

    # --- Backtest Loop ---
    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        
        if active_trade:
            # Trailing Stop Calculation
            highest_high = max(highest_high, current['high'])
            trailing_stop = highest_high - (current['atr'] * config.get('atr_mult', 3.0))
            
            ts_hit = config.get('use_ts', False) and current['low'] <= trailing_stop
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            
            if sl_hit or tp_hit or ts_hit or prev['exit_signal']:
                reason = "TS Hit" if ts_hit else ("Stop Loss" if sl_hit else ("Target Profit" if tp_hit else "Signal Exit"))
                active_trade.exit_price = current['open'] if not ts_hit else min(current['open'], trailing_stop)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None; highest_high = 0
                
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
            highest_high = current['high']
            
    return trades, df

# --- 3. UI STYLING ---
st.set_page_config(layout="wide", page_title="Pro-Tracer Pro")
st.markdown("<style>.stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; } .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; } .stat-label { color: #999; font-size: 0.85rem; } .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }</style>", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
st.sidebar.title("🎗️ Pro-Tracer Engine")
view_mode = st.sidebar.radio("View Mode", ["Main Dashboard", "Parameter Optimizer"])
st.sidebar.divider()

symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strat_choice = st.sidebar.selectbox("Select Strategy", ["PK Strategy (Positional)", "RSI 60 Cross", "EMA Ribbon", "EMA & RSI Synergy", "Double Bottom Breakout"])
selected_tf = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=4)
capital = st.sidebar.number_input("Initial Capital", value=100000.0)

# MultiIndex Fix for yfinance
@st.cache_data
def get_data(symbol, start, end, tf):
    data = yf.download(symbol, start=start, end=end, interval=tf, auto_adjust=True)
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    data.columns = [str(col).lower() for col in data.columns]
    return data

# --- 5. PAGE ROUTING ---

if view_mode == "Main Dashboard":
    start_str = st.sidebar.text_input("Start Date", value="2010-01-01")
    end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))
    
    config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 15, 'hhv_period': 20}
    st.sidebar.divider()
    use_ts = st.sidebar.toggle("Trailing Stop (ATR)", False); config['use_ts'] = use_ts
    config['atr_mult'] = st.sidebar.slider("ATR Mult", 1.0, 5.0, 3.0) if use_ts else 3.0
    use_sl = st.sidebar.toggle("Stop Loss", True); config['sl_val'] = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0; config['use_sl'] = use_sl
    use_tp = st.sidebar.toggle("Target Profit", True); config['tp_val'] = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0; config['use_tp'] = use_tp
    use_slip = st.sidebar.toggle("Slippage", True); config['slippage_val'] = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slip else 0; config['use_slippage'] = use_slip

    if st.sidebar.button("🚀 Run Backtest"):
        data = get_data(symbol, start_str, end_str, selected_tf)
        if data is not None:
            trades, processed_df = run_backtest(data, symbol, config, strat_choice)
            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                # ... (Rest of your original Dashboard logic for metrics and 4 tabs)
                st.success(f"Backtest Complete for {strat_choice}")
                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                with t1:
                    # Logic from your stable code
                    df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                    total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                    st.metric("Total Return", f"{total_ret:.2f}%")
                    st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Equity Curve"), use_container_width=True)
            else:
                st.warning("No trades found.")

else:
    # --- SEPARATE OPTIMIZATION PAGE ---
    st.header("🧪 Strategy Parameter Optimizer")
    st.info("Brute-force testing to find the best RSI or EMA levels for this specific stock.")
    
    c1, c2 = st.columns(2)
    with c1:
        param_to_opt = st.selectbox("Parameter to Optimize", ["RSI Level", "EMA Fast Period", "ATR Trailing Mult"])
        test_range = st.slider("Select Testing Range", 5, 80, (20, 60))
    with c2:
        step = st.number_input("Step Size", value=5)
    
    if st.button("Start Optimization"):
        data = get_data(symbol, "2018-01-01", date.today().strftime('%Y-%m-%d'), selected_tf)
        results = []
        
        for val in range(test_range[0], test_range[1] + 1, step):
            test_config = {'ema_fast': val if "EMA" in param_to_opt else 20, 
                           'atr_mult': val/10 if "ATR" in param_to_opt else 3.0,
                           'use_sl': True, 'sl_val': 5.0, 'use_tp': False, 'use_ts': "ATR" in param_to_opt, 'use_slippage': True, 'slippage_val': 0.1}
            
            t_list, _ = run_backtest(data.copy(), symbol, test_config, strat_choice)
            if t_list:
                final_ret = np.prod([1 + t.pnl_pct for t in t_list]) - 1
                results.append({param_to_opt: val, "Return %": final_ret * 100, "Trades": len(t_list)})
        
        res_df = pd.DataFrame(results).sort_values("Return %", ascending=False)
        st.write("### Optimization Results")
        st.dataframe(res_df, use_container_width=True)
        st.plotly_chart(px.bar(res_df, x=param_to_opt, y="Return %", title=f"Optimization results for {param_to_opt}"), use_container_width=True)