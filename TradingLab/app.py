import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, date
from dataclasses import dataclass

# Importing your custom modules
from strategies import StrategyFactory
from analytics import AnalyticsEngine
from visuals import VisualLibrary

# --- DATA STRUCTURE ---
@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime = None
    exit_price: float = None
    exit_reason: str = None
    pnl_pct: float = 0.0

# --- UI CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Institutional Strategy Lab")

st.markdown("""
    <style>
    .stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; }
    .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; }
    .stat-label { color: #999; font-size: 0.85rem; }
    .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strat_choice = st.sidebar.selectbox("Select Strategy", ["RSI 60 Cross", "EMA Ribbon"])
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_date = st.sidebar.text_input("Start Date", value="2010-01-01")

# Strategy Config
config = {'use_sl': True, 'sl_val': 5.0, 'use_tp': True, 'tp_val': 25.0, 'use_slippage': True, 'slippage_val': 0.1}
if strat_choice == "EMA Ribbon":
    config['ema_fast'] = st.sidebar.number_input("Fast EMA", 20)
    config['ema_slow'] = st.sidebar.number_input("Slow EMA", 50)
    config['ema_exit'] = st.sidebar.number_input("Exit EMA", 30)

# --- EXECUTION ---
if st.sidebar.button("🚀 Run Modular Analysis"):
    try:
        data = yf.download(symbol, start=start_date)
        if not data.empty:
            data.columns = [c.lower() for c in data.columns]
            
            # 1. Fetch Signals from StrategyFactory Module
            if strat_choice == "RSI 60 Cross":
                long_s, exit_s = StrategyFactory.rsi_60_cross(data, config)
            else:
                long_s, exit_s = StrategyFactory.ema_ribbon(data, config)
            
            # 2. Main Trade Loop
            trades = []
            active_trade = None
            slip = (config['slippage_val'] / 100) if config['use_slippage'] else 0
            
            for i in range(1, len(data)):
                if active_trade:
                    # Risk Checks
                    sl_hit = config['use_sl'] and data['low'].iloc[i] <= active_trade.entry_price * (1 - config['sl_val']/100)
                    tp_hit = config['use_tp'] and data['high'].iloc[i] >= active_trade.entry_price * (1 + config['tp_val']/100)
                    
                    if sl_hit or tp_hit or exit_s.iloc[i-1]:
                        active_trade.exit_price = data['open'].iloc[i] * (1 - slip)
                        active_trade.exit_date = data.index[i]
                        active_trade.exit_reason = "SL" if sl_hit else ("TP" if tp_hit else "Signal")
                        active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                        trades.append(active_trade)
                        active_trade = None
                elif long_s.iloc[i-1]:
                    active_trade = Trade(symbol=symbol, entry_date=data.index[i], entry_price=data['open'].iloc[i] * (1 + slip))
            
            if trades:
                # 3. Process Results
                df_t = pd.DataFrame([vars(t) for t in trades])
                df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
                duration = df_t['exit_date'].max() - df_t['entry_date'].min()
                
                # 4. Calculate Metrics via AnalyticsEngine Module
                m = AnalyticsEngine.calculate_all_metrics(df_t, capital, duration)
                
                # 5. Display 4-Tab UI
                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                
                with t1:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Returns", f"{m['total_ret']:.2f}%")
                    c2.metric("Max Drawdown", f"{m['mdd']:.2f}%")
                    c3.metric("Win Ratio", f"{m['win_rate']:.2f}%")
                    c4.metric("Total Trades", m['total_trades'])
                    st.divider()
                    st.download_button("📥 Export CSV", df_t.to_csv().encode('utf-8'), f"{symbol}_trades.csv")

                with t2:
                    cl, cr = st.columns([1, 2.5])
                    with cl:
                        with st.expander("📊 Performance", expanded=True):
                            draw_stat("CAGR", f"{m['cagr']:.2f}%")
                            draw_stat("Sharpe Ratio", f"{m['sharpe']:.2f}")
                            draw_stat("Calmar Ratio", f"{m['calmar']:.2f}")
                        with st.expander("🔥 Streaks"):
                            draw_stat("Win Streak", m['max_w_s'])
                            draw_stat("Loss Streak", m['max_l_s'])
                    with cr:
                        st.plotly_chart(px.line(df_t, x='exit_date', y='equity', title="Equity Curve", template="plotly_dark"), use_container_width=True)

                with t3:
                    # 6. Render Charts via VisualLibrary Module
                    VisualLibrary.render_charts(df_t)
                    
                
                with t4:
                    st.dataframe(df_t, use_container_width=True)
            else:
                st.warning("No trades executed with current parameters.")
    except Exception as e:
        st.error(f"Error: {e}")