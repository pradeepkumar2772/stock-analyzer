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

# --- 2. BACKTEST ENGINE ---
def run_backtest(df, symbol, config):
    trades = []
    active_trade = None
    
    # Pre-calculate Indicators
    df['ema_f'] = df['close'].ewm(span=config['ema_f'], adjust=False).mean()
    df['ema_s'] = df['close'].ewm(span=config['ema_s'], adjust=False).mean()
    
    # ATR Calculation for Trailing Stop
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

# --- 3. UI LAYOUT ---
st.set_page_config(layout="wide", page_title="Pro-Tracer Suite")
st.sidebar.title("🎗️ Pro-Tracer Suite")

symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
capital = st.sidebar.number_input("Capital", value=100000.0)

# MultiIndex Fix for yfinance
@st.cache_data
def get_data(symbol):
    data = yf.download(symbol, start="2018-01-01", auto_adjust=True)
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    data.columns = [str(col).lower() for col in data.columns]
    return data

data = get_data(symbol)

# --- THE PAGE SWITCHER ---
page = st.tabs(["📊 Performance Lab", "🧪 Optimization Engine"])

# --- PAGE 1: PERFORMANCE LAB ---
with page[0]:
    st.subheader(f"Strategy Analytics: {symbol}")
    st.sidebar.markdown("### Single Run Settings")
    e_f = st.sidebar.number_input("Fast EMA", 5, 100, 20)
    e_s = st.sidebar.number_input("Slow EMA", 20, 200, 50)
    a_m = st.sidebar.slider("ATR Multiplier", 1.0, 6.0, 3.0)
    
    if st.sidebar.button("Run Performance Lab"):
        trades = run_backtest(data.copy(), symbol, {'ema_f': e_f, 'ema_s': e_s, 'atr_mult': a_m})
        if trades:
            df_t = pd.DataFrame([vars(t) for t in trades])
            df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
            
            # Key Metrics
            total_ret = (df_t['equity'].iloc[-1] / capital - 1) * 100
            win_rate = (len(df_t[df_t['pnl_pct'] > 0]) / len(df_t)) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Net Profit", f"{total_ret:.2f}%")
            c2.metric("Win Rate", f"{win_rate:.2f}%")
            c3.metric("Total Trades", len(df_t))
            
            st.plotly_chart(px.line(df_t, x='exit_date', y='equity', title="Growth of Capital"), use_container_width=True)
            st.dataframe(df_t, use_container_width=True)
        else:
            st.warning("No trades found.")

# --- PAGE 2: OPTIMIZATION ENGINE ---
with page[1]:
    st.subheader("Brute-Force Parameter Discovery")
    st.info("Find the 'Golden Parameters' by testing thousands of combinations.")
    
    col1, col2 = st.columns(2)
    with col1:
        ema_range = st.slider("Select EMA Fast Range", 5, 60, (10, 30), step=5)
    with col2:
        mult_range = st.slider("Select ATR Mult Range", 1.5, 5.0, (2.0, 4.0), step=0.5)
        
    if st.button("🚀 Start Optimization Process"):
        results = []
        progress_bar = st.progress(0)
        
        # Define search space
        e_space = range(ema_range[0], ema_range[1] + 1, 5)
        m_space = np.arange(mult_range[0], mult_range[1] + 0.1, 0.5)
        total_steps = len(e_space) * len(m_space)
        step = 0
        
        for e in e_space:
            for m in m_space:
                t_list = run_backtest(data.copy(), symbol, {'ema_f': e, 'ema_s': 50, 'atr_mult': m})
                if t_list:
                    pnl = [t.pnl_pct for t in t_list]
                    ret = (np.prod([1 + p for p in pnl]) - 1) * 100
                    results.append({'EMA': e, 'ATR_Mult': m, 'Return_%': ret, 'Trades': len(t_list)})
                step += 1
                progress_bar.progress(step / total_steps)
        
        res_df = pd.DataFrame(results).sort_values('Return_%', ascending=False)
        st.write("### Discoveries")
        st.dataframe(res_df.style.highlight_max(axis=0, subset=['Return_%']), use_container_width=True)
        
        # Heatmap
        
        fig = px.density_heatmap(res_df, x="EMA", y="ATR_Mult", z="Return_%", 
                                 text_auto=True, title="Profitability Heatmap", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)