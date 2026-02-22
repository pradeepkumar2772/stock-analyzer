import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from dataclasses import dataclass

# IMPORT CUSTOM MODULES
from strategies import StrategyFactory
from analytics import AnalyticsEngine
from visuals import VisualLibrary

@dataclass
class Trade:
    symbol: str; entry_date: datetime; entry_price: float
    exit_date: datetime = None; exit_price: float = None; exit_reason: str = None; pnl_pct: float = 0.0

st.set_page_config(layout="wide", page_title="Institutional Strategy Lab")

# Sidebar
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strat_choice = st.sidebar.selectbox("Select Strategy", ["RSI 60 Cross", "EMA Ribbon"])
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_date = st.sidebar.text_input("Start Date", value="2010-01-01")

# Global Config
config = {'use_sl': True, 'sl_val': 5.0, 'use_tp': True, 'tp_val': 25.0, 'use_slippage': True, 'slippage_val': 0.1, 'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 30}

if st.sidebar.button("🚀 Run Modular Analysis"):
    try:
        data = yf.download(symbol, start=start_date)
        if not data.empty:
            # --- ROBUST COLUMN CLEANING ---
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(c).lower() for c in data.columns]
            # ------------------------------

            # 1. Fetch Signals
            if strat_choice == "RSI 60 Cross":
                long_s, exit_s = StrategyFactory.rsi_60_cross(data, config)
            else:
                long_s, exit_s = StrategyFactory.ema_ribbon(data, config)
            
            # 2. Trade Processing Loop
            trades = []
            active_trade = None
            for i in range(1, len(data)):
                if active_trade:
                    sl_hit = config['use_sl'] and data['low'].iloc[i] <= active_trade.entry_price * (1 - config['sl_val']/100)
                    if sl_hit or exit_s.iloc[i-1]:
                        active_trade.exit_price = data['open'].iloc[i]
                        active_trade.exit_date = data.index[i]
                        active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                        active_trade.exit_reason = "SL" if sl_hit else "Signal"
                        trades.append(active_trade); active_trade = None
                elif long_s.iloc[i-1]:
                    active_trade = Trade(symbol=symbol, entry_date=data.index[i], entry_price=data['open'].iloc[i])
            
            if trades:
                df_t = pd.DataFrame([vars(t) for t in trades])
                df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
                duration = df_t['exit_date'].max() - df_t['entry_date'].min()
                
                # 3. Use Analytics & Visuals Modules
                m = AnalyticsEngine.calculate_all_metrics(df_t, capital, duration)
                
                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                with t1:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Returns", f"{m['total_ret']:.2f}%")
                    c2.metric("Max Drawdown", f"{m['mdd']:.2f}%")
                with t3:
                    VisualLibrary.render_charts(df_t)
                    
            else:
                st.warning("No trades found.")
    except Exception as e:
        st.error(f"Error: {e}")