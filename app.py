import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date, timedelta

# --- 1. NIFTY 200 CONSTITUENTS ---
NIFTY_200 = sorted([
    "ABB.NS", "ACC.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS", 
    "ATGL.NS", "AWL.NS", "ABCAPITAL.NS", "ABFRL.NS", "ALKEM.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS", 
    "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", 
    "BAJAJFINSV.NS", "BAJAJHLDNG.NS", "BALKRISIND.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BANKINDIA.NS", "BEL.NS", 
    "BERGEPAINT.NS", "BHARATFORG.NS", "BHEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BIOCON.NS", "BOSCHLTD.NS", "BRITANNIA.NS", 
    "CGPOWER.NS", "CANBK.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", 
    "CUMMINSIND.NS", "DLF.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "DIXON.NS", "DRREDDY.NS", 
    "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "FORTIS.NS", "GAIL.NS", "GMRINFRA.NS", "GLAND.NS", 
    "GODREJCP.NS", "GODREJPROP.NS", "GRASIM.NS", "GUJGASLTD.NS", "HAL.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", 
    "HAVELLS.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", 
    "IDFCFIRSTB.NS", "ITC.NS", "INDIANB.NS", "IOC.NS", "IRCTC.NS", "IRFC.NS", "IGL.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", 
    "INFY.NS", "IPCALAB.NS", "JSWENERGY.NS", "JSWSTEEL.NS", "JINDALSTEL.NS", "JIOCINANCE.NS", "JUBLFOOD.NS", "KOTAKBANK.NS", 
    "LTIM.NS", "LT.NS", "LUPIN.NS", "M&M.NS", "M&MFIN.NS", "MARICO.NS", "MARUTI.NS", "MAXHEALTH.NS", "MAZDOCK.NS", 
    "MPHASIS.NS", "NHPC.NS", "NMDC.NS", "NTPC.NS", "NESTLEIND.NS", "NYKAA.NS", "OBEROIRLTY.NS", "ONGC.NS", "PAGEIND.NS", 
    "PATANJALI.NS", "PERSISTENT.NS", "PETRONET.NS", "PIDILITIND.NS", "POLYCAB.NS", "POONAWALLA.NS", "POWERGRID.NS", 
    "PRESTIGE.NS", "RECLTD.NS", "RELIANCE.NS", "RVNL.NS", "SBICARD.NS", "SBILIFE.NS", "SBIN.NS", "SRF.NS", "SHREECEM.NS", 
    "SHRIRAMFIN.NS", "SIEMENS.NS", "SONACOMS.NS", "SUNPHARMA.NS", "SUNTV.NS", "SUPREMEIND.NS", "TATACOMM.NS", 
    "TATACONSUM.NS", "TATAELXSI.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", 
    "TORNTPHARM.NS", "TRENT.NS", "ULTRACEMCO.NS", "UPL.NS", "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "VOLTAS.NS", "WIPRO.NS", 
    "YESBANK.NS", "ZOMATO.NS", "ZYDUSLIFE.NS"
])

# --- 2. CORE DATA STRUCTURE ---
@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime = None
    exit_price: float = None
    exit_reason: str = None
    pnl_pct: float = 0.0

# --- 3. UI STYLING ---
st.set_page_config(layout="wide", page_title="Nifty 200 Strategy Lab")
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

# --- 4. SIDEBAR ---
st.sidebar.title("🎗️ Nifty 200 Engine")
symbol = st.sidebar.selectbox("Select Scrip", NIFTY_200, index=NIFTY_200.index("BRITANNIA.NS"))
strat_choice = st.sidebar.selectbox("Strategy", ["RSI 60 Cross", "EMA Ribbon"])

col1, col2 = st.sidebar.columns(2)
with col1:
    start_dt = st.date_input("Start Date", value=date(2015, 1, 1))
with col2:
    end_dt = st.date_input("End Date", value=date.today())

capital = st.sidebar.number_input("Initial Capital", value=100000.0)
st.sidebar.divider()
config = {'use_sl': True, 'sl_val': 5.0, 'use_tp': True, 'tp_val': 25.0, 'use_slip': True, 'slip_val': 0.1}

# --- 5. EXECUTION ---
if st.sidebar.button("🚀 Run Backtest"):
    try:
        data = yf.download(symbol, start=start_dt, end=end_dt, interval="1d")
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(c).lower() for c in data.columns]
            
            if strat_choice == "RSI 60 Cross":
                data['rsi'] = ta.rsi(data['close'], length=14)
                long_s = (data['rsi'] > 60) & (data['rsi'].shift(1) <= 60)
                exit_s = (data['rsi'] < 60) & (data['rsi'].shift(1) >= 60)
            else:
                data['ema_f'] = ta.ema(data['close'], length=20)
                data['ema_s'] = ta.ema(data['close'], length=50)
                long_s = (data['ema_f'] > data['ema_s']) & (data['ema_f'].shift(1) <= data['ema_s'].shift(1))
                exit_s = (data['ema_f'] < data['ema_s']) & (data['ema_f'].shift(1) >= data['ema_s'].shift(1))

            trades = []; active = None
            slip = (config['slip_val'] / 100) if config['use_slip'] else 0
            
            for i in range(1, len(data)):
                if active:
                    sl_hit = config['use_sl'] and data['low'].iloc[i] <= active.entry_price * (1 - config['sl_val']/100)
                    tp_hit = config['use_tp'] and data['high'].iloc[i] >= active.entry_price * (1 + config['tp_val']/100)
                    if sl_hit or tp_hit or exit_s.iloc[i-1]:
                        active.exit_price = data['open'].iloc[i] * (1 - slip)
                        active.exit_date = data.index[i]
                        active.pnl_pct = (active.exit_price - active.entry_price) / active.entry_price
                        active.exit_reason = "SL" if sl_hit else ("TP" if tp_hit else "Signal")
                        trades.append(active); active = None
                elif long_s.iloc[i-1]:
                    active = Trade(symbol=symbol, entry_date=data.index[i], entry_price=data['open'].iloc[i] * (1 + slip))
            
            if trades:
                df_t = pd.DataFrame([vars(t) for t in trades])
                df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
                total_ret = (df_t['equity'].iloc[-1] / capital - 1) * 100
                peak = df_t['equity'].cummax(); dd = (df_t['equity'] - peak) / peak; mdd = dd.min() * 100
                wins = df_t[df_t['pnl_pct'] > 0]
                
                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Details"])
                with t1:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Return", f"{total_ret:.2f}%")
                    c2.metric("Max Drawdown", f"{mdd:.2f}%")
                    c3.metric("Win Ratio", f"{(len(wins)/len(df_t)*100):.1f}%")
                    c4.metric("Total Trades", len(df_t))
                with t2:
                    col_l, col_r = st.columns([1, 2.5])
                    with col_l:
                        with st.expander("📊 Performance", expanded=True):
                            draw_stat("Scrip", symbol); draw_stat("Return", f"{total_ret:.2f}%"); draw_stat("MDD", f"{mdd:.2f}%")
                    with col_r:
                        st.plotly_chart(px.line(df_t, x='exit_date', y='equity', title="Equity Growth", template="plotly_dark"), use_container_width=True)
                with t3:
                    st.plotly_chart(px.area(df_t, x='exit_date', y=dd*100, title="Underwater Drawdown (%)", color_discrete_sequence=['#e74c3c'], template="plotly_dark"), use_container_width=True)
                with t4:
                    st.dataframe(df_t, use_container_width=True)
            else:
                st.warning("No trades found in the selected range.")
    except Exception as e:
        st.error(f"Error: {e}")