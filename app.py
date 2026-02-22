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

# --- 2. MULTI-STRATEGY ENGINE (Updated with Book Logic) ---
def run_backtest(df, symbol, config, strategy_type):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    # --- Indicator Pre-calculations ---
    # Standard RSI (Used for RSI 60 and RSI Divergence Logic)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    # ATR Calculation (For ATR Band Breakout Strategy)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # HHV / LLV (For Price Breakout Strategy)
    lookback = config.get('hhv_period', 20)
    df['hhv'] = df['high'].rolling(window=lookback).max()
    df['llv'] = df['low'].rolling(window=lookback).min()

    # --- Strategy Signal Mapping ---
    if strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)
    
    elif strategy_type == "ATR Band Breakout":
        # Strategy: Buy when price crosses SMA + ATR, Exit when price breaks below SMA - ATR
        df['upper_band'] = df['sma_20'] + df['atr']
        df['lower_band'] = df['sma_20'] - df['atr']
        df['long_signal'] = (df['close'] > df['upper_band']) & (df['close'].shift(1) <= df['upper_band'].shift(1))
        df['exit_signal'] = (df['close'] < df['lower_band']) & (df['close'].shift(1) >= df['lower_band'].shift(1))

    elif strategy_type == "HHV/LLV Breakout":
        # Strategy: Buy on breakout of highest high of X periods
        df['long_signal'] = (df['close'] > df['hhv'].shift(1))
        df['exit_signal'] = (df['close'] < df['llv'].shift(1))

    else: # EMA Ribbon
        df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
        df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = (df['ema_fast'] < df['ema_exit']) & (df['ema_fast'].shift(1) >= df['ema_exit'].shift(1))

    # --- Trade Loop ---
    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            if sl_hit or tp_hit or prev['exit_signal']:
                reason = "Stop Loss" if sl_hit else ("Target Profit" if tp_hit else "Signal Exit")
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
    return trades, df

# --- 3. UI STYLING (SAME AS BASELINE) ---
st.set_page_config(layout="wide", page_title="Strategy Lab Pro")
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

# --- 4. SIDEBAR ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
# Updated dropdown with Book Strategies
strat_choice = st.sidebar.selectbox("Select Strategy", ["RSI 60 Cross", "EMA Ribbon", "ATR Band Breakout", "HHV/LLV Breakout"])

tf_map = {"1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", "1 Hour": "1h", "Daily": "1d"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=4)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2015-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

config = {}
# Context-specific sidebar inputs
if strat_choice == "EMA Ribbon":
    config['ema_fast'] = st.sidebar.number_input("Fast EMA", 20)
    config['ema_slow'] = st.sidebar.number_input("Slow EMA", 50)
    config['ema_exit'] = st.sidebar.number_input("Exit EMA", 30)
elif strat_choice == "HHV/LLV Breakout":
    config['hhv_period'] = st.sidebar.slider("Breakout Period", 5, 50, 20)

st.sidebar.divider()
use_sl = st.sidebar.toggle("Stop Loss", True); config['sl_val'] = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0; config['use_sl'] = use_sl
use_tp = st.sidebar.toggle("Target Profit", True); config['tp_val'] = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0; config['use_tp'] = use_tp
use_slip = st.sidebar.toggle("Slippage", True); config['slippage_val'] = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slip else 0; config['use_slippage'] = use_slip

# --- 5. EXECUTION (Logic remains exactly same as baseline) ---
if st.sidebar.button("🚀 Run Backtest"):
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            trades, processed_df = run_backtest(data, symbol, config, strat_choice)
            
            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
                df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                
                # Calculations & Tabs (Copied from baseline)
                wins = df_trades[df_trades['pnl_pct'] > 0]; losses = df_trades[df_trades['pnl_pct'] <= 0]
                total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                duration = df_trades['exit_date'].max() - df_trades['entry_date'].min()
                years_v = max(duration.days / 365.25, 0.1)
                cagr = (((df_trades['equity'].iloc[-1] / capital) ** (1/years_v)) - 1) * 100
                peak = df_trades['equity'].cummax(); drawdown = (df_trades['equity'] - peak) / peak; mdd = drawdown.min() * 100
                sharpe = (df_trades['pnl_pct'].mean()/df_trades['pnl_pct'].std()*np.sqrt(252)) if len(df_trades)>1 else 0.0
                rr = (wins['pnl_pct'].mean()/abs(losses['pnl_pct'].mean())) if not losses.empty else 0.0
                exp = (total_ret/len(df_trades))
                calmar = abs(cagr/mdd) if mdd != 0 else 0.0
                pnl_b = (df_trades['pnl_pct'] > 0).astype(int); strk = pnl_b.groupby((pnl_b != pnl_b.shift()).cumsum()).cumcount() + 1
                max_w_s = strk[pnl_b == 1].max() if not wins.empty else 0; max_l_s = strk[pnl_b == 0].max() if not losses.empty else 0

                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                
                with t1:
                    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                    r1c1.metric("Total Returns (%)", f"{total_ret:.2f}%"); r1c2.metric("Max Drawdown(MDD)", f"{mdd:.2f}%"); r1c3.metric("Win Ratio", f"{(len(wins)/len(df_trades)*100):.2f}%"); r1c4.metric("Total Trades", len(df_trades))
                    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                    r2c1.metric("Initial Capital", f"{capital:,.2f}"); r2c2.metric("Final Capital", f"{df_trades['equity'].iloc[-1]:,.2f}"); r2c3.metric("CAGR", f"{cagr:.2f}%"); r2c4.metric("Avg Return/Trade", f"{(df_trades['pnl_pct'].mean()*100):.2f}%")
                    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
                    r3c1.metric("Risk-Reward Ratio", f"{rr:.2f}"); r3c2.metric("Expectancy", f"{exp:.2f}"); r3c3.metric("Sharpe Ratio", f"{sharpe:.2f}"); r3c4.metric("Calmer Ratio", f"{calmar:.2f}")

                with t3:
                    st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Equity Curve", color_discrete_sequence=['#3498db']), use_container_width=True)
                    st.plotly_chart(px.area(df_trades, x='exit_date', y=drawdown*100, title="Underwater Drawdown (%)", color_discrete_sequence=['#e74c3c']), use_container_width=True)

                with t4:
                    st.dataframe(df_trades[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl_pct', 'exit_reason']], use_container_width=True)
            else:
                st.warning("No trades executed.")
    except Exception as e:
        st.error(f"Execution Error: {e}")