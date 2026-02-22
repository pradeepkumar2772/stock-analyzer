import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date, timedelta

# --- 1. THE ULTIMATE NSE INDEX REPOSITORY ---
SECTORS = {
    "Nifty Auto": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "HEROMOTOCO.NS", "ASHOKLEY.NS", "BHARATFORG.NS", "BALKRISIND.NS"],
    "Nifty Bank": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "AUBANK.NS", "FEDERALBNK.NS"],
    "Nifty Cement": ["ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "SHREECEM.NS", "ACC.NS", "DALBHARAT.NS", "JKCEMENT.NS", "RAMCOCEM.NS", "INDIACEM.NS"],
    "Nifty Chemicals": ["PIDILITIND.NS", "SRF.NS", "UPL.NS", "SOLARINDS.NS", "COROMANDEL.NS", "PIIND.NS", "NAVINFLUOR.NS", "TATACHEM.NS", "DEEPAKNTR.NS"],
    "Nifty Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS"],
    "Nifty Financial Services Ex-Bank": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS", "MUTHOOTFIN.NS", "PFC.NS", "RECLTD.NS", "SBI CARD.NS", "HDFCLIFE.NS", "SBILIFE.NS"],
    "Nifty FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "VBL.NS", "GODREJCP.NS", "DABUR.NS", "TATACONSUM.NS", "MARICO.NS", "COLPAL.NS"],
    "Nifty Healthcare": ["SUNPHARMA.NS", "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "MAXHEALTH.NS", "LUPIN.NS", "FORTIS.NS", "TORNTPHARM.NS"],
    "Nifty IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "LTIM.NS", "TECHM.NS", "PERSISTENT.NS", "MPHASIS.NS", "COFORGE.NS"],
    "Nifty Media": ["SUNTV.NS", "ZEEL.NS", "PVRINOX.NS", "NAZARA.NS", "NETWORK18.NS", "TV18BRDCST.NS", "SAREGAMA.NS", "DBCORP.NS"],
    "Nifty Metal": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "JINDALSTEL.NS", "VEDL.NS", "NMDC.NS", "SAIL.NS", "NATIONALUM.NS"],
    "Nifty PSU Bank": ["SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS", "INDIANB.NS", "BANKINDIA.NS", "MAHABANK.NS"],
    "Nifty Realty": ["DLF.NS", "LODHA.NS", "GODREJPROP.NS", "PHOENIXLTD.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "BRIGADE.NS", "SOBHA.NS"],
    "Nifty Consumer Durables": ["TITAN.NS", "DIXON.NS", "HAVELLS.NS", "VOLTAS.NS", "BLUESTARCO.NS", "CROMPTON.NS", "KALYANKJIL.NS", "AMBER.NS"],
    "Nifty Oil and Gas": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "HINDPETRO.NS", "OIL.NS", "PETRONET.NS", "IGL.NS"],
    "Nifty EV & New Age Auto": ["RELIANCE.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "TATAMOTORS.NS", "ASHOKLEY.NS"],
    "Nifty India Defence": ["BEL.NS", "HAL.NS", "SOLARINDS.NS", "MAZDOCK.NS", "BHARATFORG.NS", "BDL.NS", "COCHINSHIP.NS", "GRSE.NS", "DATAPATTNS.NS"],
    "Nifty India Railways PSU": ["IRFC.NS", "RVNL.NS", "IRCTC.NS", "CONCOR.NS", "RAILTEL.NS", "RITES.NS", "IRCON.NS"],
    "Nifty India Tourism": ["INDIGO.NS", "INDHOTEL.NS", "IRCTC.NS", "JUBLFOOD.NS", "EIHOTEL.NS", "CHALET.NS", "DEVYANI.NS", "LEMONTREE.NS"],
    "Nifty CPSE": ["NTPC.NS", "ONGC.NS", "POWERGRID.NS", "COALINDIA.NS", "BEL.NS", "BPCL.NS", "OIL.NS", "NHPC.NS", "SJVN.NS"],
    "Nifty Infrastructure": ["LT.NS", "RELIANCE.NS", "BHARTIARTL.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS", "ADANIPORTS.NS", "GRASIM.NS"],
    "Nifty PSE": ["NTPC.NS", "ONGC.NS", "BEL.NS", "HAL.NS", "POWERGRID.NS", "COALINDIA.NS", "IOC.NS", "BPCL.NS", "IRFC.NS", "PFC.NS"],
    "Nifty India Digital": ["BHARTIARTL.NS", "TCS.NS", "INFY.NS", "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS", "POLICYBZR.NS", "TATCOMM.NS"],
    "Nifty SME EMERGE": ["NSE-SME-1.NS", "NSE-SME-2.NS"] # Update with specific SME Tickers if needed
}

@dataclass
class Trade:
    symbol: str; direction: str; entry_date: datetime; entry_price: float
    exit_date: datetime = None; exit_price: float = None; exit_reason: str = None; pnl_pct: float = 0.0

# --- 2. THE ENGINE ---
def run_scan(df, symbol, strat):
    # Flatten MultiIndex and Lowercase
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    
    if strat == "RSI 60 Cross":
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)
    else: # EMA Ribbon
        df['ema_f'] = ta.ema(df['close'], length=20)
        df['ema_s'] = ta.ema(df['close'], length=50)
        df['long_signal'] = (df['ema_f'] > df['ema_s']) & (df['ema_f'].shift(1) <= df['ema_s'].shift(1))
        df['exit_signal'] = (df['ema_f'] < df['ema_s']) & (df['ema_f'].shift(1) >= df['ema_s'].shift(1))

    trades = []
    active = None
    for i in range(1, len(df)):
        if active:
            if df['low'].iloc[i] <= active.entry_price * 0.95 or df['exit_signal'].iloc[i-1]:
                active.exit_price = df['open'].iloc[i]
                active.pnl_pct = (active.exit_price - active.entry_price) / active.entry_price
                trades.append(active); active = None
        elif df['long_signal'].iloc[i-1]:
            active = Trade(symbol=symbol, direction="Long", entry_date=df.index[i], entry_price=df['open'].iloc[i])
    return trades, active

# --- 3. UI ---
st.set_page_config(layout="wide", page_title="Ultimate NSE Scanner")
st.sidebar.title("🎗️ NSE Index Scanner")
mode = st.sidebar.radio("Navigation", ["Single Stock", "Multi-Index Scanner"])
strat_choice = st.sidebar.selectbox("Strategy", ["RSI 60 Cross", "EMA Ribbon"])

if mode == "Multi-Index Scanner":
    selected_idx = st.sidebar.selectbox("Choose Nifty Index", sorted(list(SECTORS.keys())))
    period = st.sidebar.selectbox("Period", ["1 Year", "2 Years", "5 Years"])
    
    if st.sidebar.button("🔍 Execute Scan"):
        results = []
        bar = st.progress(0)
        stocks = SECTORS[selected_idx]
        
        for i, sym in enumerate(stocks):
            bar.progress((i+1)/len(stocks))
            try:
                data = yf.download(sym, period="2y", interval="1d", progress=False)
                if not data.empty:
                    trades, active = run_scan(data, sym, strat_choice)
                    if trades:
                        df_t = pd.DataFrame([vars(t) for t in trades])
                        df_t['equity'] = 1000 * (1 + df_t['pnl_pct']).cumprod()
                        ret = (df_t['equity'].iloc[-1] / 1000 - 1) * 100
                        wr = (len(df_t[df_t['pnl_pct'] > 0]) / len(df_t)) * 100
                        results.append({"Stock": sym, "Return %": round(ret, 2), "Win Rate %": round(wr, 2), "Status": "🟢 Long" if active else "⚪ Wait"})
            except: continue
        
        if results:
            res_df = pd.DataFrame(results).sort_values("Return %", ascending=False)
            st.subheader(f"Leaderboard: {selected_idx}")
            st.table(res_df)
            
            
            
            fig = px.bar(res_df, x="Stock", y="Return %", color="Return %", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)