import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
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

# --- 2. UI STYLING ---
st.set_page_config(layout="wide", page_title="Institutional Strategy Lab")
st.markdown("""
    <style>
    .stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; }
    .report-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .report-table td { border: 1px solid #2d2f3b; padding: 10px; text-align: center; color: #fff; font-size: 0.85rem; }
    .profit { background-color: #1b5e20 !important; color: #c8e6c9 !important; font-weight: bold; }
    .loss { background-color: #b71c1c !important; color: #ffcdd2 !important; font-weight: bold; }
    .total-cell { font-weight: bold; color: #fff; background-color: #1e3a5f !important; }
    .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; }
    .stat-label { color: #999; font-size: 0.85rem; }
    .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- 3. SIDEBAR ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strat_choice = st.sidebar.selectbox("Select Strategy", ["RSI 60 Cross", "EMA Ribbon"])

tf_map = {"1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", "1 Hour": "1h", "Daily": "1d"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=4)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_date = st.sidebar.text_input("Start Date", value="2010-01-01")

config = {'use_sl': True, 'sl_val': 5.0, 'use_tp': True, 'tp_val': 25.0, 'use_slippage': True, 'slippage_val': 0.1}
if strat_choice == "EMA Ribbon":
    config['ema_fast'] = st.sidebar.number_input("Fast EMA", 20)
    config['ema_slow'] = st.sidebar.number_input("Slow EMA", 50)
    config['ema_exit'] = st.sidebar.number_input("Exit EMA", 30)

# --- 4. EXECUTION ---
if st.sidebar.button("🚀 Run Backtest"):
    try:
        data = yf.download(symbol, start=start_date, interval=tf_map[selected_tf])
        if not data.empty:
            # FIX: Robust Column Cleaning (Prevents 'tuple' error)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(c).lower() for c in data.columns]
            
            # Indicators using pandas-ta
            if strat_choice == "RSI 60 Cross":
                data['rsi'] = ta.rsi(data['close'], length=14)
                long_signal = (data['rsi'] > 60) & (data['rsi'].shift(1) <= 60)
                exit_signal = (data['rsi'] < 60) & (data['rsi'].shift(1) >= 60)
            else:
                data['ema_f'] = ta.ema(data['close'], length=config.get('ema_fast', 20))
                data['ema_s'] = ta.ema(data['close'], length=config.get('ema_slow', 50))
                data['ema_e'] = ta.ema(data['close'], length=config.get('ema_exit', 30))
                long_signal = (data['ema_f'] > data['ema_s']) & (data['ema_f'].shift(1) <= data['ema_s'].shift(1))
                exit_signal = (data['ema_f'] < data['ema_e']) & (data['ema_f'].shift(1) >= data['ema_e'].shift(1))
            
            # Trade Loop
            trades = []
            active_trade = None
            slip = (config['slippage_val'] / 100) if config['use_slippage'] else 0
            
            for i in range(1, len(data)):
                if active_trade:
                    sl_hit = config['use_sl'] and data['low'].iloc[i] <= active_trade.entry_price * (1 - config['sl_val']/100)
                    tp_hit = config['use_tp'] and data['high'].iloc[i] >= active_trade.entry_price * (1 + config['tp_val']/100)
                    
                    if sl_hit or tp_hit or exit_signal.iloc[i-1]:
                        active_trade.exit_price = data['open'].iloc[i] * (1 - slip)
                        active_trade.exit_date = data.index[i]
                        active_trade.exit_reason = "SL" if sl_hit else ("TP" if tp_hit else "Signal")
                        active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                        trades.append(active_trade)
                        active_trade = None
                elif long_signal.iloc[i-1]:
                    active_trade = Trade(symbol=symbol, entry_date=data.index[i], entry_price=data['open'].iloc[i] * (1 + slip))
            
            if trades:
                df_t = pd.DataFrame([vars(t) for t in trades])
                df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
                
                # Math Metrics
                wins = df_t[df_t['pnl_pct'] > 0]; losses = df_t[df_t['pnl_pct'] <= 0]
                total_ret = (df_t['equity'].iloc[-1] / capital - 1) * 100
                duration = df_t['exit_date'].max() - df_t['entry_date'].min()
                years = max(duration.days / 365.25, 0.1)
                cagr = (((df_t['equity'].iloc[-1] / capital) ** (1/years)) - 1) * 100
                peak = df_t['equity'].cummax(); drawdown = (df_t['equity'] - peak) / peak; mdd = drawdown.min() * 100
                sharpe = (df_t['pnl_pct'].mean() / df_t['pnl_pct'].std() * np.sqrt(252)) if len(df_t) > 1 else 0
                
                # UI Tabs
                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                
                with t1:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Returns", f"{total_ret:.2f}%")
                    c2.metric("Max Drawdown", f"{mdd:.2f}%")
                    c3.metric("Win Ratio", f"{(len(wins)/len(df_t)*100):.2f}%")
                    c4.metric("Total Trades", len(df_t))
                    st.divider()
                    st.download_button("📥 Export CSV", df_t.to_csv(index=False).encode('utf-8'), f"{symbol}_trades.csv")

                with t2:
                    col_l, col_r = st.columns([1, 2.5])
                    with col_l:
                        with st.expander("📊 Returns", expanded=True):
                            draw_stat("CAGR", f"{cagr:.2f}%")
                            draw_stat("Sharpe", f"{sharpe:.2f}")
                        with st.expander("📉 Drawdown"):
                            draw_stat("Max Drawdown", f"{mdd:.2f}%")
                    with col_r:
                        st.plotly_chart(px.line(df_t, x='exit_date', y='equity', title="Equity Curve", template="plotly_dark"), use_container_width=True)
                        

                with t3:
                    # 4 Charts Vertical Stack
                    yearly = df_t.groupby(df_t['exit_date'].dt.year)['pnl_pct'].sum() * 100
                    st.plotly_chart(go.Figure(data=[go.Bar(x=yearly.index, y=yearly.values, marker_color='#3498db', text=yearly.values.round(1), textposition='outside')]).update_layout(title="Return by Period (%)", template="plotly_dark"), use_container_width=True)
                    
                    exits = df_t['exit_reason'].value_counts()
                    st.plotly_chart(go.Figure(data=[go.Bar(x=exits.index, y=exits.values, marker_color='#2ecc71', text=exits.values, textposition='outside')]).update_layout(title="Exits Distribution", template="plotly_dark"), use_container_width=True)

                with t4:
                    st.dataframe(df_t, use_container_width=True)
            else:
                st.warning("No trades executed.")
    except Exception as e:
        st.error(f"Error: {e}")