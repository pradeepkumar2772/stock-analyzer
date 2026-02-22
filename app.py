import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    # Indicators
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + df['close'].diff().fillna(0).rolling(14).apply(lambda x: x[x>0].sum()/abs(x[x<0].sum()) if x[x<0].sum() != 0 else 10).mean()))
    
    # Entry Logic with Filter
    df['long_signal'] = (df['ema20'] > df['ema50']) & (df['ema20'].shift(1) <= df['ema50'].shift(1))
    if config['use_rsi_filter']:
        df['long_signal'] &= (df['rsi'] > 50)
        
    df['exit_signal'] = (df['ema20'] < df['ema30']) & (df['ema20'].shift(1) >= df['ema30'].shift(1))

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            
            if sl_hit or tp_hit or prev['exit_signal']:
                reason = "Stop Loss" if sl_hit else ("Target Profit" if tp_hit else "EMA Cross Exit")
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None
        elif prev['long_signal']:
            entry_p = current['open'] * (1 + slippage)
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=entry_p)
    return trades, df

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="PK Ribbon Strategy Lab")
st.sidebar.title("🎗️ PK Ribbon Engine")
symbol = st.sidebar.text_input("Symbol", value="RELIANCE.NS").upper()

tf_limits = {"1 Day": "1d", "1 Hour": "1h", "15 Minutes": "15m", "5 Minutes": "5m"}
selected_tf_label = st.sidebar.selectbox("Select Timeframe", list(tf_limits.keys()), index=0)
capital = st.sidebar.number_input("Initial Capital", value=100000)

start_str = st.sidebar.text_input("Start Date (YYYY-MM-DD)", value="2020-01-01")
end_str = st.sidebar.text_input("End Date (YYYY-MM-DD)", value=date.today().strftime('%Y-%m-%d'))

st.sidebar.divider()
st.sidebar.subheader("⚙️ Strategy Enhancements")
use_rsi_filter = st.sidebar.checkbox("Apply RSI Filter (>50)", value=False)
use_sl = st.sidebar.checkbox("Enable Stop Loss", value=True)
sl_val = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0
use_tp = st.sidebar.checkbox("Enable Target Profit", value=True)
tp_val = st.sidebar.slider("Target %", 1.0, 100.0, 25.0) if use_tp else 0
use_slippage = st.sidebar.checkbox("Apply Slippage", value=True)
slippage_val = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slippage else 0

if st.sidebar.button("🚀 Run Full Strategy Lab"):
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_limits[selected_tf_label], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            
            config = {'use_sl': use_sl, 'sl_val': sl_val, 'use_tp': use_tp, 'tp_val': tp_val, 'use_slippage': use_slippage, 'slippage_val': slippage_val, 'capital': capital, 'use_rsi_filter': use_rsi_filter}
            trades, processed_df = run_backtest(data, symbol, config)

            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                
                # --- SCOREBOARD ---
                st.subheader("📊 Performance Scoreboard")
                c1, c2, c3, c4 = st.columns(4)
                total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                c1.metric("Total Return (%)", f"{total_ret:.2f}%")
                c2.metric("Win Rate", f"{(len(df_trades[df_trades['pnl_pct']>0])/len(df_trades)*100):.1f}%")
                
                peak = df_trades['equity'].cummax()
                mdd = ((df_trades['equity'] - peak) / peak).min() * 100
                c3.metric("Max Drawdown", f"{mdd:.2f}%")
                
                recovery_factor = abs(total_ret / mdd) if mdd != 0 else 0
                c4.metric("Recovery Factor", f"{recovery_factor:.2f}")

                # --- MONTHLY HEATMAP ---
                st.divider()
                st.subheader("🗓️ Monthly Performance Heatmap")
                df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
                df_trades['month'] = df_trades['exit_date'].dt.strftime('%b')
                df_trades['year'] = df_trades['exit_date'].dt.year
                heatmap_data = df_trades.groupby(['year', 'month'])['pnl_pct'].sum().unstack().fillna(0) * 100
                st.dataframe(heatmap_data.style.background_gradient(cmap='RdYlGn', axis=None))
                

                # --- CHARTS WITH VISUAL MARKERS ---
                st.divider()
                st.subheader("🕯️ Technical Audit & Visual Signals")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=processed_df.index, open=processed_df['open'], high=processed_df['high'], low=processed_df['low'], close=processed_df['close'], name="Price"), row=1, col=1)
                
                # Entry Markers
                fig.add_trace(go.Scatter(x=df_trades['entry_date'], y=df_trades['entry_price'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00'), name="Buy Signal"), row=1, col=1)
                # Exit Markers
                fig.add_trace(go.Scatter(x=df_trades['exit_date'], y=df_trades['exit_price'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000'), name="Sell Signal"), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df_trades['exit_date'], y=df_trades['equity'], name="Equity Curve", fill='tozeroy', line=dict(color='#00ffcc')), row=2, col=1)
                fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(df_trades[['entry_date', 'exit_date', 'exit_reason', 'pnl_pct']])
            else:
                st.warning("No trades found.")
    except Exception as e:
        st.error(f"Error: {e}")