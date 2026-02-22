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
    
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    df['long_signal'] = (df['ema20'] > df['ema50']) & (df['ema20'].shift(1) <= df['ema50'].shift(1))
    df['exit_signal'] = (df['ema20'] < df['ema30']) & (df['ema20'].shift(1) >= df['ema30'].shift(1))

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        if active_trade:
            sl_hit, tp_hit = False, False
            if config['use_sl']:
                sl_p = active_trade.entry_price * (1 - config['sl_val'] / 100)
                sl_hit = current['low'] <= sl_p
            if config['use_tp']:
                tp_p = active_trade.entry_price * (1 + config['tp_val'] / 100)
                tp_hit = current['high'] >= tp_p
            indicator_exit = prev['exit_signal']

            if sl_hit or tp_hit or indicator_exit:
                reason = "Stop Loss" if sl_hit else ("Target" if tp_hit else "EMA Cross Exit")
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
st.set_page_config(layout="wide", page_title="PK Ribbon Performance Pro")

st.sidebar.title("🎗️ PK Ribbon Engine")
symbol = st.sidebar.text_input("Symbol", value="RELIANCE.NS")

tf_limits = {
    "1 Minute": {"val": "1m", "max_days": 7},
    "5 Minutes": {"val": "5m", "max_days": 59},
    "15 Minutes": {"val": "15m", "max_days": 59},
    "1 Hour": {"val": "1h", "max_days": 729},
    "1 Day": {"val": "1d", "max_days": 20000},
}

selected_tf_label = st.sidebar.selectbox("Select Timeframe", list(tf_limits.keys()), index=4)
selected_tf = tf_limits[selected_tf_label]["val"]
max_days_allowed = tf_limits[selected_tf_label]["max_days"]

capital = st.sidebar.number_input("Initial Capital", value=100000)

# DATE RANGE INPUTS (Stable version logic)
fifty_years_ago = date.today() - timedelta(days=50*365)
user_start = st.sidebar.date_input("Start Date", value=date(2020, 1, 1), min_value=fifty_years_ago)
user_end = st.sidebar.date_input("End Date", value=date.today())

st.sidebar.divider()
st.sidebar.subheader("⚙️ Toggles")
use_sl = st.sidebar.checkbox("Enable Stop Loss", value=True)
sl_val = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0
use_tp = st.sidebar.checkbox("Enable Target Profit", value=True)
tp_val = st.sidebar.slider("Target %", 1.0, 100.0, 25.0) if use_tp else 0
use_slippage = st.sidebar.checkbox("Apply Slippage", value=True)
slippage_val = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slippage else 0

if st.sidebar.button("🚀 Run Analysis"):
    earliest_allowed = date.today() - timedelta(days=max_days_allowed)
    final_start = user_start if user_start >= earliest_allowed else earliest_allowed
    
    if user_start < earliest_allowed:
        st.info(f"💡 **Auto-Corrected:** Start date adjusted to {final_start} for {selected_tf_label}.")

    if final_start >= user_end:
        st.error("❌ End date must be after the start date.")
    else:
        try:
            with st.spinner(f'Fetching data and calculating metrics...'):
                data = yf.download(symbol, start=final_start, end=user_end, interval=selected_tf, auto_adjust=True)
                
                if data.empty:
                    st.error("No data returned.")
                else:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    data.columns = [str(col).lower() for col in data.columns]
                    data = data.dropna()
                    
                    config = {'use_sl': use_sl, 'sl_val': sl_val, 'use_tp': use_tp, 'tp_val': tp_val, 'use_slippage': use_slippage, 'slippage_val': slippage_val, 'capital': capital}
                    trades, processed_df = run_backtest(data, symbol, config)

                    if not trades:
                        st.warning("No trades generated.")
                    else:
                        df_trades = pd.DataFrame([vars(t) for t in trades])
                        
                        # --- PERFORMANCE ENHANCEMENTS ---
                        total_ret_pct = (df_trades['pnl_pct'] + 1).prod() - 1
                        wins = df_trades[df_trades['pnl_pct'] > 0]
                        losses = df_trades[df_trades['pnl_pct'] <= 0]
                        win_rate = (len(wins) / len(df_trades)) * 100
                        
                        # CAGR
                        days_diff = (processed_df.index[-1] - processed_df.index[0]).days
                        years = days_diff / 365.25 if days_diff > 0 else 1
                        cagr = (((capital * (1 + total_ret_pct)) / capital) ** (1 / years) - 1) * 100
                        
                        # Risk-Reward & Profit Factor
                        avg_win = wins['pnl_pct'].mean() if not wins.empty else 0
                        avg_loss = abs(losses['pnl_pct'].mean()) if not losses.empty else 0.0001
                        risk_reward = avg_win / avg_loss
                        
                        total_profit = wins['pnl_pct'].sum()
                        total_loss = abs(losses['pnl_pct'].sum())
                        profit_factor = total_profit / total_loss if total_loss != 0 else total_profit
                        
                        # Expectancy
                        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
                        
                        # Max Drawdown
                        df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                        rolling_max = df_trades['equity'].cummax()
                        drawdowns = (df_trades['equity'] - rolling_max) / rolling_max
                        max_dd = drawdowns.min() * 100

                        # --- DASHBOARD UI ---
                        st.subheader("📊 Professional Strategy Scorecard")
                        
                        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                        r1c1.metric("Total Return", f"{total_ret_pct*100:.2f}%")
                        r1c2.metric("CAGR", f"{cagr:.2f}%")
                        r1c3.metric("Success Ratio", f"{win_rate:.1f}%")
                        r1c4.metric("Max Drawdown", f"{max_dd:.2f}%")

                        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                        r2c1.metric("Profit Factor", f"{profit_factor:.2f}")
                        r2c2.metric("Risk:Reward", f"1:{risk_reward:.2f}")
                        r2c3.metric("Expectancy", f"{expectancy*100:.2f}%")
                        r2c4.metric("Avg Return/Trade", f"{df_trades['pnl_pct'].mean()*100:.2f}%")

                        # Charts
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                        fig.add_trace(go.Candlestick(x=processed_df.index, open=processed_df['open'], high=processed_df['high'], low=processed_df['low'], close=processed_df['close'], name="Price"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['ema20'], name="EMA 20", line=dict(color='yellow', width=1)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['ema50'], name="EMA 50", line=dict(color='red', width=1)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df_trades['exit_date'], y=df_trades['equity'], name="Equity Curve", line=dict(color='#00ffcc')), row=2, col=1)
                        
                        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # Download Button
                        st.divider()
                        csv = df_trades.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Trade Log", data=csv, file_name=f"{symbol}_audit.csv", mime='text/csv')
                        st.dataframe(df_trades)
        except Exception as e:
            st.error(f"Error: {e}")