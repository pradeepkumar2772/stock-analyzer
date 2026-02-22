import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from datetime import datetime

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
    pnl_value: float = 0.0

# --- 2. BACKTEST ENGINE ---
def run_backtest(df, symbol, config):
    trades = []
    active_trade = None
    slippage = config['slippage'] / 100
    
    # Simple Strategy: EMA Crossover
    df['ema_fast'] = df['close'].ewm(span=config['ema_fast']).mean()
    df['ema_slow'] = df['close'].ewm(span=config['ema_slow']).mean()
    df['signal'] = np.where(df['ema_fast'] > df['ema_slow'], 1, 0)

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # EXIT LOGIC (Check today's movement for trades opened previously)
        if active_trade:
            active_trade.pnl_value = (current['close'] - active_trade.entry_price) * (config['capital'] / active_trade.entry_price)
            
            price_low = current['low']
            price_high = current['high']
            
            sl_price = active_trade.entry_price * (1 - config['sl'] / 100)
            tp_price = active_trade.entry_price * (1 + config['tp'] / 100)
            
            sl_hit = price_low <= sl_price
            tp_hit = price_high >= tp_price
            signal_exit = prev['signal'] == 0

            if sl_hit or tp_hit or signal_exit:
                reason = "Stop Loss" if sl_hit else ("Target" if tp_hit else "Signal")
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None

        # ENTRY LOGIC (If signal yesterday, enter on today's open)
        elif prev['signal'] == 1:
            entry_p = current['open'] * (1 + slippage)
            active_trade = Trade(
                symbol=symbol,
                direction="Long",
                entry_date=current.name,
                entry_price=entry_p
            )
            
    return trades, df

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Pro Backtester v2.0")

st.sidebar.title("🛠 Engine Settings")
symbol = st.sidebar.text_input("Symbol (e.g. RELIANCE.NS, TSLA)", value="RELIANCE.NS")
capital = st.sidebar.number_input("Initial Capital", value=100000)
st.sidebar.divider()
sl_pct = st.sidebar.slider("Stop Loss %", 0.5, 10.0, 2.0)
tp_pct = st.sidebar.slider("Target Profit %", 1.0, 20.0, 5.0)
slippage_pct = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1)
st.sidebar.divider()
ema_f = st.sidebar.number_input("Fast EMA", value=9)
ema_s = st.sidebar.number_input("Slow EMA", value=21)

if st.sidebar.button("🚀 Run Backtest"):
    try:
        with st.spinner('Syncing with Market Data...'):
            # FIXED DATA FETCHING
            raw_data = yf.download(symbol, period="2y", interval="1d")
            
            if raw_data.empty:
                st.error("No data found. Please check the symbol.")
            else:
                data = raw_data.copy()
                # Fix for yfinance MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                data.columns = [str(col).lower() for col in data.columns]
                data = data.dropna()
                
                config = {
                    'sl': sl_pct, 'tp': tp_pct, 'slippage': slippage_pct, 
                    'ema_fast': ema_f, 'ema_slow': ema_s, 'capital': capital
                }
                
                trades, processed_df = run_backtest(data, symbol, config)

                if not trades:
                    st.warning("Strategy generated 0 trades. Try widening your SL/Target.")
                else:
                    # --- DASHBOARD ---
                    df_trades = pd.DataFrame([vars(t) for t in trades])
                    win_rate = (len(df_trades[df_trades['pnl_pct'] > 0]) / len(df_trades)) * 100
                    total_ret = df_trades['pnl_pct'].sum() * 100
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Trades", len(df_trades))
                    c2.metric("Win Rate", f"{win_rate:.1f}%")
                    c3.metric("Total Return", f"{total_ret:.1f}%")
                    c4.metric("Avg Trade", f"{df_trades['pnl_pct'].mean()*100:.2f}%")

                    # --- PLOTTING ---
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                        vertical_spacing=0.05, row_heights=[0.7, 0.3])
                    
                    fig.add_trace(go.Candlestick(x=processed_df.index, 
                                                 open=processed_df['open'], high=processed_df['high'], 
                                                 low=processed_df['low'], close=processed_df['close'], 
                                                 name="Price"), row=1, col=1)
                    
                    # Entry Markers
                    fig.add_trace(go.Scatter(x=df_trades['entry_date'], y=df_trades['entry_price'],
                                             mode='markers', marker=dict(symbol='triangle-up', size=10, color='lime'),
                                             name='Buy Signal'), row=1, col=1)

                    # Equity Curve
                    df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                    fig.add_trace(go.Scatter(x=df_trades['exit_date'], y=df_trades['equity'], 
                                             name="Equity Curve", line=dict(color='cyan', width=2)), row=2, col=1)
                    
                    fig.update_layout(height=700, template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # --- LOGS ---
                    st.subheader("📊 Trade History")
                    st.dataframe(df_trades[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'exit_reason', 'pnl_pct']], 
                                 use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")