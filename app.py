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

# --- 2. BACKTEST ENGINE WITH TRAILING STOP ---
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
    
    highest_high_since_entry = 0

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        if active_trade:
            # Update Chandelier Exit level
            highest_high_since_entry = max(highest_high_since_entry, curr['high'])
            trailing_stop = highest_high_since_entry - (curr['atr'] * config['atr_mult'])
            
            # Exit Logic
            ema_cross_exit = prev['ema_f'] < prev['ema_s']
            ts_hit = curr['low'] <= trailing_stop
            
            if ema_cross_exit or ts_hit:
                active_trade.exit_price = curr['open']
                active_trade.exit_date = curr.name
                active_trade.exit_reason = "TS Hit" if ts_hit else "EMA Cross"
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None
                highest_high_since_entry = 0
                
        elif prev['ema_f'] > prev['ema_s']: # Entry
            active_trade = Trade(symbol=symbol, entry_date=curr.name, entry_price=curr['open'])
            highest_high_since_entry = curr['high']
            
    return trades

# --- 3. UI SETUP ---
st.set_page_config(layout="wide", page_title="Pro-Tracer Optimizer")
st.sidebar.title("🛠️ Pro-Tracer Engine")

symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
mode = st.sidebar.selectbox("Mode", ["Single Backtest", "Parameter Optimization"])
capital = st.sidebar.number_input("Initial Capital", value=100000.0)

# --- 4. DATA FETCHING (With MultiIndex Fix) ---
@st.cache_data
def get_data(symbol):
    data = yf.download(symbol, start="2015-01-01", auto_adjust=True)
    if data.empty: return None
    # Fix for newer yfinance MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(col).lower() for col in data.columns]
    return data

data = get_data(symbol)

# --- 5. EXECUTION LOGIC ---
if data is not None:
    if mode == "Parameter Optimization":
        st.header(f"🔍 Optimization: {symbol}")
        col1, col2 = st.columns(2)
        with col1: ema_range = st.slider("EMA Fast Range", 5, 50, (10, 25), step=5)
        with col2: mult_range = st.slider("ATR Multiplier Range", 1.5, 5.0, (2.0, 4.0), step=0.5)
        
        if st.button("Run Optimization"):
            results = []
            for e_f in range(ema_range[0], ema_range[1] + 1, 5):
                for m in np.arange(mult_range[0], mult_range[1] + 0.1, 0.5):
                    conf = {'ema_f': e_f, 'ema_s': 50, 'atr_mult': m}
                    trades = run_backtest(data.copy(), symbol, conf)
                    if trades:
                        pnl_list = [t.pnl_pct for t in trades]
                        total_ret = (np.prod([1 + p for p in pnl_list]) - 1) * 100
                        results.append({'EMA_F': e_f, 'ATR_Mult': m, 'Return_%': total_ret, 'Trades': len(trades)})
            
            res_df = pd.DataFrame(results).sort_values('Return_%', ascending=False)
            st.write("### Optimization Results (Sorted by Return)")
            st.dataframe(res_df.style.highlight_max(axis=0, subset=['Return_%']), use_container_width=True)
            
            fig = px.density_heatmap(res_df, x="EMA_F", y="ATR_Mult", z="Return_%", 
                                     title="Optimization Heatmap", text_auto=True, color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Single Backtest UI (Original Style)
        st.sidebar.divider()
        e_f = st.sidebar.number_input("EMA Fast", value=20)
        atr_m = st.sidebar.slider("ATR Trailing Multiplier", 1.0, 6.0, 3.0)
        
        if st.sidebar.button("Run Backtest"):
            conf = {'ema_f': e_f, 'ema_s': 50, 'atr_mult': atr_m}
            trades = run_backtest(data.copy(), symbol, conf)
            
            if trades:
                df_t = pd.DataFrame([vars(t) for t in trades])
                df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
                
                # Metrics
                total_ret = (df_t['equity'].iloc[-1] / capital - 1) * 100
                win_rate = (len(df_t[df_t['pnl_pct'] > 0]) / len(df_t)) * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Return", f"{total_ret:.2f}%")
                m2.metric("Win Rate", f"{win_rate:.2f}%")
                m3.metric("Total Trades", len(df_t))
                
                st.plotly_chart(px.line(df_t, x='exit_date', y='equity', title="Equity Curve"), use_container_width=True)
                st.write("### Trade Log")
                st.dataframe(df_t, use_container_width=True)
            else:
                st.warning("No trades found for these parameters.")
else:
    st.error("Could not fetch data. Check the symbol.")