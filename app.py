import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date, timedelta

# --- 1. CORE DATA STRUCTURES ---
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

# --- 2. BACKTEST ENGINE ---
def run_backtest(df, symbol, config):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema30'] = df['close'].ewm(span=30, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    df['long_signal'] = (df['ema20'] > df['ema50']) & (df['ema20'].shift(1) <= df['ema50'].shift(1))
    
    if config['use_rsi_filter']:
        mode = config['rsi_mode']
        if mode == "Greater Than": df['long_signal'] &= (df['rsi'] > config['rsi_val1'])
        elif mode == "Less Than": df['long_signal'] &= (df['rsi'] < config['rsi_val1'])
        elif mode == "Between Range": df['long_signal'] &= (df['rsi'] >= config['rsi_val1']) & (df['rsi'] <= config['rsi_val2'])
        
    df['exit_signal'] = (df['ema20'] < df['ema30']) & (df['ema20'].shift(1) >= df['ema30'].shift(1))

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        if active_trade:
            sl_hit = False
            if config['use_sl']:
                sl_hit = current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            
            tp_hit = False
            if config['use_tp']:
                tp_hit = current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            
            if sl_hit or tp_hit or prev['exit_signal']:
                reason = "Stop Loss" if sl_hit else ("Target Profit" if tp_hit else "System Builder")
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
    return trades, df

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Institutional Backtesting Report")

st.markdown("""
    <style>
    .stMetric { background-color: #1a1c24; padding: 20px; border-radius: 8px; border: 1px solid #2d2f3b; }
    .report-table { width: 100%; border-collapse: collapse; background-color: transparent; margin-top: 10px; }
    .report-table th { background-color: #2d2f3b; color: #bbb; padding: 12px; border: 1px solid #2d2f3b; font-size: 0.85rem; text-transform: uppercase; }
    .report-table td { border: 1px solid #2d2f3b; padding: 10px; text-align: center; color: #fff; font-size: 0.9rem; }
    .profit { background-color: #1b5e20 !important; color: #c8e6c9 !important; font-weight: bold; text-shadow: 1px 1px 2px #000; }
    .loss { background-color: #b71c1c !important; color: #ffcdd2 !important; font-weight: bold; text-shadow: 1px 1px 2px #000; }
    .total-cell { font-weight: bold; color: #fff; background-color: #1e3a5f !important; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🎗️ PK Ribbon Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
tf_limits = {"Daily": "1d", "1 Hour": "1h", "15 Minutes": "15m", "5 Minutes": "5m"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_limits.keys()))
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2005-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

st.sidebar.divider()
st.sidebar.subheader("🛡️ RSI & Risk Settings")
use_rsi_filter = st.sidebar.toggle("Enable RSI Filter", value=False)
rsi_mode, rsi_val1, rsi_val2 = "Greater Than", 50.0, 70.0
if use_rsi_filter:
    rsi_mode = st.sidebar.selectbox("RSI Mode", ["Greater Than", "Less Than", "Between Range"])
    rsi_val1 = st.sidebar.number_input("RSI 1", 0.0, 100.0, 50.0)
    if rsi_mode == "Between Range": rsi_val2 = st.sidebar.number_input("RSI 2", 0.0, 100.0, 70.0)

use_sl = st.sidebar.toggle("Enable Stop Loss", value=True)
sl_val = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0
use_tp = st.sidebar.toggle("Enable Target Profit", value=True)
tp_val = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0
use_slippage = st.sidebar.toggle("Apply Slippage", value=True)
slippage_val = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slippage else 0

if st.sidebar.button("🚀 Generate Full Report"):
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_limits[selected_tf], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            
            config = {'use_sl': use_sl, 'sl_val': sl_val, 'use_tp': use_tp, 'tp_val': tp_val, 'use_slippage': use_slippage, 'slippage_val': slippage_val, 'capital': capital, 'use_rsi_filter': use_rsi_filter, 'rsi_mode': rsi_mode, 'rsi_val1': rsi_val1, 'rsi_val2': rsi_val2}
            trades, processed_df = run_backtest(data, symbol, config)

            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                
                # Metrics
                wins = df_trades[df_trades['pnl_pct'] > 0]; losses = df_trades[df_trades['pnl_pct'] <= 0]
                total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                years = max((df_trades['exit_date'].max() - pd.to_datetime(df_trades['entry_date'].min())).days / 365.25, 0.1)
                cagr = (((df_trades['equity'].iloc[-1] / capital) ** (1/years)) - 1) * 100
                peak = df_trades['equity'].cummax(); drawdown = (df_trades['equity'] - peak) / peak; mdd = drawdown.min() * 100

                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                
                with t1:
                    m1, m2, m3 = st.columns(3); m1.metric("Total Returns (%)", f"{total_ret:.2f}%"); m2.metric("Max Drawdown(MDD)", f"{mdd:.2f}%"); m3.metric("Total Trades", len(df_trades))
                    m4, m5, m6 = st.columns(3); m4.metric("Initial Capital", f"{capital:,.2f}"); m5.metric("Final Capital", f"{df_trades['equity'].iloc[-1]:,.2f}"); m6.metric("Win Ratio", f"{(len(wins)/len(df_trades)*100):.2f}%")
                    st.divider()
                    st.subheader("Monthly Returns")
                    df_trades['year'] = df_trades['exit_date'].dt.year
                    df_trades['month'] = df_trades['exit_date'].dt.strftime('%b')
                    pivot = df_trades.groupby(['year', 'month'])['pnl_pct'].sum().unstack().fillna(0) * 100
                    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns])
                    pivot['Total'] = pivot.sum(axis=1)

                    html = "<table class='report-table'><thead><tr><th>Year</th>" + "".join([f"<th>{m}</th>" for m in pivot.columns]) + "</tr></thead><tbody>"
                    for year, row in pivot.iloc[::-1].iterrows():
                        html += f"<tr><td>{year}</td>"
                        for col_name, val in row.items():
                            cls = "profit" if val > 0 else ("loss" if val < 0 else "")
                            if col_name == "Total": cls = "total-cell"
                            display_val = f"{val:.2f}%" if val != 0 else "-"
                            html += f"<td class='{cls}'>{display_val}</td>"
                        html += "</tr>"
                    html += "</tbody></table>"
                    st.markdown(html, unsafe_allow_html=True)

                with t2:
                    st.subheader("Performance & Analytics")
                    col_stats, col_charts = st.columns([1, 2.5])
                    with col_stats:
                        st.write(f"**Highest Return:** {df_trades['pnl_pct'].max()*100:.2f}%")
                        st.write(f"**Lowest Return:** {df_trades['pnl_pct'].min()*100:.2f}%")
                        st.write(f"**Avg Win:** {wins['pnl_pct'].mean()*100:.2f}%")
                        st.write(f"**Avg Loss:** {losses['pnl_pct'].mean()*100:.2f}%")
                    with col_charts:
                        st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Equity Curve Strategy", color_discrete_sequence=['#3498db']), use_container_width=True)
                        st.plotly_chart(px.area(df_trades, x='exit_date', y=drawdown*100, title="Drawdown (Long)", color_discrete_sequence=['#e74c3c']), use_container_width=True)

                with t3:
                    # --- REARRANGED VERTICAL CHARTS ---
                    st.subheader("1. Return by Period")
                    yearly_ret = df_trades.groupby('year')['pnl_pct'].sum() * 100
                    fig_period = px.bar(yearly_ret, text_auto='.1f', color_discrete_sequence=['#3498db'])
                    fig_period.update_layout(xaxis_title="", yaxis_title="Values", template="plotly_dark", height=500)
                    st.plotly_chart(fig_period, use_container_width=True)
                    st.divider()

                    st.subheader("2. Winners/Losers (Yearly)")
                    win_loss_by_year = df_trades.assign(is_win=df_trades['pnl_pct'] > 0).groupby(['year', 'is_win']).size().unstack(fill_value=0)
                    win_loss_by_year.columns = ['Losers', 'Winners']
                    win_loss_by_year['Total'] = win_loss_by_year['Winners'] + win_loss_by_year['Losers']
                    fig_wl = go.Figure()
                    fig_wl.add_trace(go.Bar(x=win_loss_by_year.index, y=win_loss_by_year['Total'], name='Total', marker_color='#3498db'))
                    fig_wl.add_trace(go.Bar(x=win_loss_by_year.index, y=win_loss_by_year['Winners'], name='Winners', marker_color='#2ecc71'))
                    fig_wl.add_trace(go.Bar(x=win_loss_by_year.index, y=win_loss_by_year['Losers'], name='Losers', marker_color='#e74c3c'))
                    fig_wl.update_layout(barmode='group', template="plotly_dark", yaxis_title="Values", height=500)
                    st.plotly_chart(fig_wl, use_container_width=True)
                    st.divider()

                    st.subheader("3. Exit Distribution")
                    exit_stats = df_trades['exit_reason'].value_counts(normalize=True) * 100
                    fig_exits = px.bar(exit_stats, text_auto='.0f', color_discrete_sequence=['#3498db'])
                    fig_exits.update_layout(showlegend=False, yaxis_title="Values", template="plotly_dark", height=400)
                    st.plotly_chart(fig_exits, use_container_width=True)
                    st.divider()

                    st.subheader("4. Trades by Day of the Week")
                    df_trades['day'] = df_trades['exit_date'].dt.day_name()
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    day_data = df_trades.assign(is_win=df_trades['pnl_pct'] > 0).groupby(['day', 'is_win']).size().unstack(fill_value=0)
                    day_data.columns = ['Losers', 'Winners']
                    day_data['Total'] = day_data['Winners'] + day_data['Losers']
                    day_data = day_data.reindex(day_order)
                    fig_day = go.Figure()
                    fig_day.add_trace(go.Bar(x=day_data.index, y=day_data['Total'], name='Total', marker_color='#3498db'))
                    fig_day.add_trace(go.Bar(x=day_data.index, y=day_data['Winners'], name='Winners', marker_color='#2ecc71'))
                    fig_day.add_trace(go.Bar(x=day_data.index, y=day_data['Losers'], name='Losers', marker_color='#e74c3c'))
                    fig_day.update_layout(barmode='group', template="plotly_dark", yaxis_title="Values", height=500)
                    st.plotly_chart(fig_day, use_container_width=True)

                with t4:
                    st.subheader("Trade Audit Log")
                    st.dataframe(df_trades[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl_pct', 'exit_reason']], use_container_width=True)

            else:
                st.warning("No trades found.")
    except Exception as e:
        st.error(f"Error: {e}")