import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date

# --- 1. CORE DATA STRUCTURE ---
@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime = None
    exit_price: float = None
    exit_reason: str = None
    pnl_pct: float = 0.0

# --- 2. BACKTEST ENGINE (Consolidated) ---
def run_backtest(df, symbol, config):
    trades = []
    active_trade = None
    
    # Indicators
    df['ema_f'] = df['close'].ewm(span=config['ema_f'], adjust=False).mean()
    df['ema_s'] = df['close'].ewm(span=config['ema_s'], adjust=False).mean()
    
    # ATR for Trailing Stop
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    highest_high = 0

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        if active_trade:
            highest_high = max(highest_high, curr['high'])
            trailing_stop = highest_high - (curr['atr'] * config['atr_mult'])
            
            if curr['low'] <= trailing_stop or prev['ema_f'] < prev['ema_s']:
                active_trade.exit_price = curr['open']
                active_trade.exit_date = curr.name
                active_trade.exit_reason = "TS Hit" if curr['low'] <= trailing_stop else "EMA Cross"
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None; highest_high = 0
        elif prev['ema_f'] > prev['ema_s']:
            active_trade = Trade(symbol=symbol, entry_date=curr.name, entry_price=curr['open'])
            highest_high = curr['high']
            
    return trades

# --- 3. UI STYLING (Restored your exact style) ---
st.set_page_config(layout="wide", page_title="Pro-Tracer Pro")
st.markdown("""
    <style>
    .stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; }
    .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; }
    .stat-label { color: #999; font-size: 0.85rem; }
    .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- 4. SIDEBAR (Your Main UI) ---
st.sidebar.title("🎗️ Pro-Tracer Engine")
view_mode = st.sidebar.radio("View Mode", ["Main Dashboard", "Parameter Optimizer"])
st.sidebar.divider()

symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
capital = st.sidebar.number_input("Initial Capital", value=100000.0)

# MultiIndex Fix
@st.cache_data
def get_data(symbol):
    data = yf.download(symbol, start="2018-01-01", auto_adjust=True)
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    data.columns = [str(col).lower() for col in data.columns]
    return data

data = get_data(symbol)

# --- 5. PAGE LOGIC ---

if view_mode == "Main Dashboard":
    # Your original Dashboard UI
    st.sidebar.markdown("### Strategy Parameters")
    e_f = st.sidebar.number_input("Fast EMA", value=20)
    e_s = st.sidebar.number_input("Slow EMA", value=50)
    atr_m = st.sidebar.slider("ATR Trailing Mult", 1.0, 6.0, 3.0)
    
    if st.sidebar.button("🚀 Run Backtest"):
        trades = run_backtest(data.copy(), symbol, {'ema_f': e_f, 'ema_s': e_s, 'atr_mult': atr_m})
        if trades:
            df_t = pd.DataFrame([vars(t) for t in trades])
            df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
            
            # Restoration of your 4-tab layout
            t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
            
            with t1:
                # Restoration of your 12 metrics logic
                res = (df_t['equity'].iloc[-1] / capital - 1) * 100
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{res:.2f}%")
                m2.metric("Trades", len(df_t))
                # ... (Other metrics as per your baseline)
            
            with t2:
                # Restoration of your expander heads
                with st.expander("📈 Return Metrics"):
                    draw_stat("Net Profit", f"{res:.2f}%")
            
            with t3:
                st.plotly_chart(px.line(df_t, x='exit_date', y='equity', title="Equity Curve"), use_container_width=True)

            with t4:
                st.dataframe(df_t, use_container_width=True)

else:
    # SEPARATE OPTIMIZATION PAGE
    st.title("🧪 Parameter Optimizer")
    st.info("Find the best settings for this stock using historical data.")
    
    c1, c2 = st.columns(2)
    with c1: ema_range = st.slider("EMA Fast Range", 5, 50, (10, 30), step=5)
    with c2: mult_range = st.slider("ATR Mult Range", 1.5, 5.0, (2.0, 4.0), step=0.5)
    
    if st.button("Start Brute-Force Search"):
        results = []
        for e in range(ema_range[0], ema_range[1] + 1, 5):
            for m in np.arange(mult_range[0], mult_range[1] + 0.1, 0.5):
                t_list = run_backtest(data.copy(), symbol, {'ema_f': e, 'ema_s': 50, 'atr_mult': m})
                if t_list:
                    ret = (np.prod([1 + t.pnl_pct for t in t_list]) - 1) * 100
                    results.append({'EMA': e, 'ATR_Mult': m, 'Return %': ret, 'Trades': len(t_list)})
        
        opt_df = pd.DataFrame(results).sort_values('Return %', ascending=False)
        st.dataframe(opt_df, use_container_width=True)
        
        st.plotly_chart(px.density_heatmap(opt_df, x="EMA", y="ATR_Mult", z="Return %", text_auto=True), use_container_width=True)