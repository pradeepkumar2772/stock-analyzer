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
    
    # EMAs for PK Strategy & Ribbon
    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    # ATR for Trailing Stop (Chandelier Exit)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # HHV for Trailing Stop reference
    highest_high = 0

    # Pattern/Breakout Logic
    h_per = config.get('hhv_period', 20)
    df['hhv'] = df['high'].rolling(window=h_per).max()
    df['llv'] = df['low'].rolling(window=h_per).min()
    df['neckline'] = df['high'].rolling(window=20).max()

    # --- Strategy Signals ---
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20'].shift(1)) & (df['close'] > df['ema_20'])
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
            # Update Trailing Stop level
            highest_high = max(highest_high, current['high'])
            trailing_stop_price = highest_high - (current['atr'] * config.get('atr_mult', 3.0))
            
            ts_hit = config.get('use_ts', False) and current['low'] <= trailing_stop_price
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            
            if sl_hit or tp_hit or ts_hit or prev['exit_signal']:
                reason = "TS Hit" if ts_hit else ("Stop Loss" if sl_hit else ("Target Profit" if tp_hit else "Signal Exit"))
                # If TS hits, we exit at the stop price or open, whichever is worse
                active_trade.exit_price = min(current['open'], trailing_stop_price) if ts_hit else current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None; highest_high = 0
                
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
            highest_high = current['high']
            
    return trades, df

# --- 3. UI STYLING ---
st.set_page_config(layout="wide", page_title="Strategy Lab Pro")
st.markdown("<style>.stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; } .report-table { width: 100%; border-collapse: collapse; margin-top: 10px; } .report-table td { border: 1px solid #2d2f3b; padding: 10px; text-align: center; color: #fff; font-size: 0.85rem; } .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; } .stat-label { color: #999; font-size: 0.85rem; } .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }</style>", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
# Added PK Strategy to your selectbox
strat_choice = st.sidebar.selectbox("Select Strategy", ["PK Strategy (Positional)", "RSI 60 Cross", "EMA Ribbon", "EMA & RSI Synergy", "Double Bottom Breakout"])
tf_map = {"1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", "1 Hour": "1h", "Daily": "1d"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=4)
capital = st.sidebar.number_input("Initial Capital", value=100000.0)
start_str = st.sidebar.text_input("Start Date", value="2010-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 15, 'hhv_period': 20}
st.sidebar.divider()

# Added Trailing Stop Controls
use_ts = st.sidebar.toggle("ATR Trailing Stop", False); config['use_ts'] = use_ts
config['atr_mult'] = st.sidebar.slider("ATR Multiplier", 1.0, 5.0, 3.0) if use_ts else 3.0

use_sl = st.sidebar.toggle("Stop Loss", True); config['sl_val'] = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0; config['use_sl'] = use_sl
use_tp = st.sidebar.toggle("Target Profit", True); config['tp_val'] = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0; config['use_tp'] = use_tp
use_slip = st.sidebar.toggle("Slippage", True); config['slippage_val'] = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slip else 0; config['use_slippage'] = use_slip

# --- 5. EXECUTION ---
if st.sidebar.button("🚀 Run Backtest"):
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
        if not data.empty:
            # Flatten MultiIndex Columns
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            
            trades, processed_df = run_backtest(data, symbol, config, strat_choice)
            
            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date']); df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                df_trades['year'] = df_trades['exit_date'].dt.year; df_trades['month'] = df_trades['exit_date'].dt.strftime('%b')
                
                wins = df_trades[df_trades['pnl_pct'] > 0]; losses = df_trades[df_trades['pnl_pct'] <= 0]
                total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                mdd = ((df_trades['equity'] - df_trades['equity'].cummax()) / df_trades['equity'].cummax()).min() * 100

                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                with t1:
                    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                    r1c1.metric("Total Return", f"{total_ret:.2f}%"); r1c2.metric("Max DD", f"{mdd:.2f}%"); r1c3.metric("Win Ratio", f"{(len(wins)/len(df_trades)*100):.2f}%"); r1c4.metric("Trades", len(df_trades))
                    st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Equity Curve"), use_container_width=True)

                with t2:
                    cl, cr = st.columns([1, 2.5])
                    with cl:
                        with st.expander("📊 Performance", expanded=True):
                            draw_stat("Net Profit", f"{total_ret:.2f}%")
                            draw_stat("Avg Win", f"{wins['pnl_pct'].mean()*100:.2f}%")
                            draw_stat("Avg Loss", f"{losses['pnl_pct'].mean()*100:.2f}%")
                        with st.expander("⏱️ Holding Period"):
                            df_trades['hold'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.days
                            draw_stat("Avg Hold", f"{df_trades['hold'].mean():.2f} days")
                    with cr:
                        st.plotly_chart(px.area(df_trades, x='exit_date', y=(df_trades['equity'] - df_trades['equity'].cummax()) / df_trades['equity'].cummax() * 100, title="Drawdown (%)", color_discrete_sequence=['#e74c3c']), use_container_width=True)

                with t4:
                    st.dataframe(df_trades, use_container_width=True)
            else:
                st.warning("No trades found.")
    except Exception as e:
        st.error(f"Execution Error: {e}")