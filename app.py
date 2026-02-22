import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date, timedelta

# --- 1. THE COMPLETE NIFTY INDEX REPOSITORY ---
# Mapped with top constituents for each specific index provided in your list
SECTORS = {
    "Nifty Auto Index": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "TVSMOTOR.NS"],
    "Nifty Bank Index": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BANKBARODA.NS"],
    "Nifty Cement Index": ["ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "SHREECEM.NS", "ACC.NS", "JKCEMENT.NS"],
    "Nifty Chemicals": ["PIDILITIND.NS", "SRF.NS", "UPL.NS", "TATACHEM.NS", "DEEPAKNTR.NS", "NAVINFLUOR.NS"],
    "Nifty Financial Services Index": ["BAJFINANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS"],
    "Nifty Financial Services 25/50 Index": ["BAJFINANCE.NS", "RECLTD.NS", "PFC.NS", "MUTHOOTFIN.NS", "HDFCLIFE.NS"],
    "Nifty Financial Services Ex-Bank index": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "PFC.NS"],
    "Nifty FMCG Index": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "VBL.NS", "GODREJCP.NS"],
    "Nifty Healthcare Index": ["SUNPHARMA.NS", "APOLLOHOSP.NS", "MAXHEALTH.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS"],
    "Nifty IT Index": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "LTIM.NS", "TECHM.NS"],
    "Nifty Media Index": ["SUNTV.NS", "ZEEL.NS", "PVRINOX.NS", "NAZARA.NS", "TV18BRDCST.NS"],
    "Nifty Metal Index": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "JINDALSTEL.NS", "VEDL.NS", "SAIL.NS"],
    "Nifty Pharma Index": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "MANKIND.NS", "LUPIN.NS", "AUROPHARMA.NS"],
    "Nifty Private Bank Index": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "FEDERALBNK.NS"],
    "Nifty PSU Bank Index": ["SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS", "INDIANB.NS"],
    "Nifty Realty Index": ["DLF.NS", "LODHA.NS", "GODREJPROP.NS", "PHOENIXLTD.NS", "OBEROIRLTY.NS", "PRESTIGE.NS"],
    "Nifty Consumer Durables Index": ["TITAN.NS", "DIXON.NS", "HAVELLS.NS", "VOLTAS.NS", "KALYANKJIL.NS", "BLUESTARCO.NS"],
    "Nifty Oil and Gas Index": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "OIL.NS"],
    "Nifty500 Healthcare": ["SUNPHARMA.NS", "MAXHEALTH.NS", "LUPIN.NS", "GLAND.NS", "SYNGENE.NS"],
    "Nifty MidSmall Financial Services Index": ["RECLTD.NS", "PFC.NS", "IDFCFIRSTB.NS", "AUBANK.NS", "M&MFIN.NS"],
    "Nifty MidSmall Healthcare Index": ["METROPOLIS.NS", "LALPATHLAB.NS", "ASTERDM.NS", "NARAYANA.NS"],
    "Nifty MidSmall IT & Telecom Index": ["PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "KPITTECH.NS", "TATACOMM.NS"],
    "Nifty Capital Markets": ["HDFCAMC.NS", "CDSL.NS", "MCX.NS", "BSE.NS", "CAMS.NS", "ISEC.NS"],
    "Nifty Commodities Index": ["RELIANCE.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ULTRACEMCO.NS", "ONGC.NS"],
    "Nifty Conglomerate 50": ["RELIANCE.NS", "LT.NS", "ADANIENT.NS", "GRASIM.NS", "M&M.NS"],
    "Nifty Core Housing Index": ["DLF.NS", "ULTRACEMCO.NS", "HDFCBANK.NS", "ASIANPAINT.NS", "HAVELLS.NS"],
    "Nifty CPSE Index": ["NTPC.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "BEL.NS", "NHPC.NS"],
    "Nifty EV & New Age Automotive Index": ["TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "SONACOMS.NS", "EXIDEIND.NS"],
    "Nifty Energy Index": ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS", "ADANIGREEN.NS"],
    "Nifty Housing Index": ["DLF.NS", "GODREJPROP.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "KOTAKBANK.NS"],
    "Nifty India Consumption Index": ["ITC.NS", "HINDUNILVR.NS", "MARUTI.NS", "TITAN.NS", "ASIANPAINT.NS"],
    "Nifty India Defence": ["BEL.NS", "HAL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "BHARATFORG.NS", "BDL.NS"],
    "Nifty India Digital": ["BHARTIARTL.NS", "TCS.NS", "INFY.NS", "ZOMATO.NS", "PAYTM.NS"],
    "Nifty India Infrastructure & Logistics": ["LT.NS", "ADANIPORTS.NS", "CONCOR.NS", "BLUESTARCO.NS", "GMRINFRA.NS"],
    "Nifty India Internet": ["ZOMATO.NS", "INFOE.NS", "NYKAA.NS", "PAYTM.NS", "DELHIVERY.NS"],
    "Nifty India Manufacturing Index": ["RELIANCE.NS", "TATASTEEL.NS", "M&M.NS", "MARUTI.NS", "JSWSTEEL.NS"],
    "Nifty India New Age Consumption": ["ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "DEVYANI.NS", "SAPPHIRE.NS"],
    "Nifty India Railways PSU": ["IRFC.NS", "RVNL.NS", "IRCON.NS", "IRCTC.NS", "RAILTEL.NS", "RITES.NS"],
    "Nifty India Tourism": ["INDHOTEL.NS", "INDIGO.NS", "EIHOTEL.NS", "CHALET.NS", "LEMONTREE.NS"],
    "Nifty Infrastructure Index": ["RELIANCE.NS", "LT.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS"],
    "Nifty IPO": ["ZOMATO.NS", "NYKAA.NS", "POLICYBZR.NS", "MAPMYINDIA.NS", "ADANIVILMAR.NS"],
    "Nifty Midcap Liquid 15 Index": ["RECLTD.NS", "PFC.NS", "YESBANK.NS", "IDFCFIRSTB.NS", "POLYCAB.NS"],
    "Nifty MidSmall India Consumption Index": ["VBL.NS", "METROBRAND.NS", "TRENT.NS", "DEVYANI.NS"],
    "Nifty MNC Index": ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "COLPAL.NS", "ABB.NS", "SIEMENS.NS"],
    "Nifty Mobility": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "INDIGO.NS", "ADANIPORTS.NS"],
    "Nifty PSE Index": ["NTPC.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "RECLTD.NS", "HAL.NS"],
    "Nifty REITs & InvITs Index": ["EMBASSY.NS", "MINDSPACE.NS", "BIREIT.NS", "PGINVIT.NS"],
    "Nifty Rural": ["M&M.NS", "HEROMOTOCO.NS", "DABUR.NS", "HINDUNILVR.NS", "ESCORTS.NS"],
    "Nifty Non-Cyclical Consumer Index": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Nifty Services Sector Index": ["HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS", "INFY.NS", "AXISBANK.NS"],
    "Nifty Shariah 25 Index": ["INFY.NS", "TCS.NS", "HINDUNILVR.NS", "HCLTECH.NS", "ASIANPAINT.NS"],
    "Nifty100 Liquid 15 Index": ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS"],
    "Nifty50 Shariah Index": ["INFY.NS", "TCS.NS", "HINDUNILVR.NS", "HCLTECH.NS", "ASIANPAINT.NS"],
    "Nifty500 Shariah Index": ["INFY.NS", "TCS.NS", "HINDUNILVR.NS", "HCLTECH.NS", "ASIANPAINT.NS"],
    "Nifty500 Multicap India Manufacturing 50:30:20 Index": ["RELIANCE.NS", "TATASTEEL.NS", "M&M.NS"],
    "Nifty500 Multicap Infrastructure 50:30:20 Index": ["LT.NS", "RELIANCE.NS", "ULTRACEMCO.NS"],
    "Nifty SME EMERGE": ["NSE-SME-TOP.NS"], # Representative of top liquid SME stocks
    "Nifty Transportation & Logistics": ["ADANIPORTS.NS", "INDIGO.NS", "TATAMOTORS.NS", "MARUTI.NS", "CONCOR.NS"]
}

@dataclass
class Trade:
    symbol: str; direction: str; entry_date: datetime; entry_price: float
    exit_date: datetime = None; exit_price: float = None; exit_reason: str = None; pnl_pct: float = 0.0

# --- 2. MULTI-STRATEGY ENGINE ---
def run_scanner_logic(df, symbol, strat_type):
    # Robust Column Flattening
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    
    # Calculate Indicators
    if strat_type == "RSI 60 Cross":
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
            # Fixed 5% Stop Loss for Scanner
            if df['low'].iloc[i] <= active.entry_price * 0.95 or df['exit_signal'].iloc[i-1]:
                active.exit_price = df['open'].iloc[i]
                active.exit_date = df.index[i]
                active.pnl_pct = (active.exit_price - active.entry_price) / active.entry_price
                trades.append(active); active = None
        elif df['long_signal'].iloc[i-1]:
            active = Trade(symbol=symbol, direction="Long", entry_date=df.index[i], entry_price=df['open'].iloc[i])
    return trades, active

# --- 3. UI DASHBOARD ---
st.set_page_config(layout="wide", page_title="Ultimate Nifty Index Scanner")

st.sidebar.title("🎗️ Global Nifty Scanner")
mode = st.sidebar.radio("Navigation", ["Single Stock View", "Multi-Index Scanner"])
strat_choice = st.sidebar.selectbox("Select Strategy", ["RSI 60 Cross", "EMA Ribbon"])

if mode == "Multi-Index Scanner":
    selected_idx = st.sidebar.selectbox("Choose Nifty Index", sorted(list(SECTORS.keys())))
    scan_period = st.sidebar.select_slider("Backtest Lookback", options=["1 Year", "2 Years", "3 Years"], value="1 Year")
    
    if st.sidebar.button("🔍 Execute Full Scan"):
        results = []
        bar = st.progress(0)
        stocks = SECTORS[selected_idx]
        
        for idx, sym in enumerate(stocks):
            bar.progress((idx+1)/len(stocks))
            try:
                data = yf.download(sym, period="3y", interval="1d", progress=False)
                if not data.empty:
                    trades, active = run_scanner_logic(data, sym, strat_choice)
                    if trades:
                        df_t = pd.DataFrame([vars(t) for t in trades])
                        df_t['equity'] = 1000 * (1 + df_t['pnl_pct']).cumprod()
                        ret = (df_t['equity'].iloc[-1] / 1000 - 1) * 100
                        wr = (len(df_t[df_t['pnl_pct'] > 0]) / len(df_t)) * 100
                        results.append({
                            "Stock": sym, 
                            "Return %": round(ret, 2), 
                            "Win Rate %": round(wr, 2), 
                            "Total Trades": len(df_t),
                            "Status": "🟢 Long" if active else "⚪ Wait"
                        })
            except: continue
        
        if results:
            res_df = pd.DataFrame(results).sort_values("Return %", ascending=False)
            st.subheader(f"Leaderboard: {selected_idx} - {scan_period}")
            
            # Key Summary Stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Top Gainer", res_df.iloc[0]['Stock'], f"{res_df.iloc[0]['Return %']}%")
            c2.metric("Avg Returns", f"{round(res_df['Return %'].mean(), 2)}%")
            c3.metric("Currently Active", len(res_df[res_df['Status'] == "🟢 Long"]))
            
            
            
            # Chart and Table
            fig = px.bar(res_df, x="Stock", y="Return %", color="Return %", color_continuous_scale="RdYlGn", title=f"Strategy Return Benchmarking")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(res_df, use_container_width=True)
        else:
            st.warning("No data found for the selected constituents.")
else:
    st.info("Please switch to 'Multi-Index Scanner' in the sidebar to view all 58 Nifty indices.")