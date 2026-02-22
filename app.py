import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date, timedelta

# --- 1. FULL NSE SECTOR DEFINITIONS ---
SECTORS = {
    "Nifty 50": ["ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LTIM.NS", "LT.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "UPL.NS", "ULTRACEMCO.NS", "WIPRO.NS"],
    "Nifty Bank": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "AUBANK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "PNB.NS", "BANDHANBNK.NS"],
    "Nifty IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "LTIM.NS", "TECHM.NS", "PERSISTENT.NS", "MPHASIS.NS", "COFORGE.NS", "LTTS.NS"],
    "Nifty FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "VBL.NS", "GODREJCP.NS", "DABUR.NS", "TATACONSUM.NS", "MARICO.NS", "COLPAL.NS"],
    "Nifty Auto": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BHARATFORG.NS", "BALKRISIND.NS"],
    "Nifty Metal": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "JINDALSTEL.NS", "VEDL.NS", "NMDC.NS", "SAIL.NS", "NATIONALUM.NS", "APLAPOLLO.NS", "RATNAMANI.NS"],
    "Nifty Pharma": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "MANKIND.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS", "LUPIN.NS", "AUROPHARMA.NS", "ALKEM.NS"]
}

@dataclass
class Trade:
    symbol: str; direction: str; entry_date: datetime; entry_price: float
    exit_date: datetime = None; exit_price: float = None; exit_reason: str = None; pnl_pct: float = 0.0

# --- 2. BACKTEST ENGINE ---
def run_scanner_backtest(df, symbol, config, strategy_type):
    trades = []
    active_trade = None
    
    # Calculate Indicators
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    df['ema_f'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_s'] = df['close'].ewm(span=50, adjust=False).mean()
    
    if strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)
    else: # EMA Ribbon
        df['long_signal'] = (df['ema_f'] > df['ema_s']) & (df['ema_f'].shift(1) <= df['ema_s'].shift(1))
        df['exit_signal'] = (df['ema_f'] < df['ema_s']) & (df['ema_f'].shift(1) >= df['ema_s'].shift(1))

    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        if active_trade:
            sl_hit = current['low'] <= active_trade.entry_price * (0.95) # 5% SL
            if sl_hit or prev['exit_signal']:
                active_trade.exit_price = current['open']
                active_trade.exit_date = current.name
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'])
            
    return trades, active_trade # Return active_trade to show current status

# --- 3. UI DASHBOARD ---
st.set_page_config(layout="wide", page_title="NSE Sector Scanner")
st.sidebar.title("🎗️ NSE Scanner Pro")
mode = st.sidebar.radio("App Mode", ["Single Stock Analysis", "Multi-Sector Scanner"])
strat_choice = st.sidebar.selectbox("Strategy Logic", ["RSI 60 Cross", "EMA Ribbon"])

if mode == "Multi-Sector Scanner":
    selected_sector = st.sidebar.selectbox("Choose NSE Sector", list(SECTORS.keys()))
    period = st.sidebar.selectbox("Backtest Period", ["6 Months", "1 Year", "2 Years"], index=1)
    
    if st.sidebar.button("🔍 Run Sector Scan"):
        st.header(f"Results for {selected_sector} ({strat_choice})")
        
        results = []
        days_map = {"6 Months": 180, "1 Year": 365, "2 Years": 730}
        start_dt = (datetime.now() - timedelta(days=days_map[period])).strftime('%Y-%m-%d')
        
        progress_text = st.empty()
        bar = st.progress(0)
        
        for idx, sym in enumerate(SECTORS[selected_sector]):
            progress_text.text(f"Analyzing {sym}...")
            bar.progress((idx + 1) / len(SECTORS[selected_sector]))
            
            data = yf.download(sym, start=start_dt, progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                data.columns = [str(c).lower() for c in data.columns]
                
                trades, active_trade = run_scanner_backtest(data, sym, {}, strat_choice)
                
                if trades:
                    df_t = pd.DataFrame([vars(t) for t in trades])
                    ret = (1 + df_t['pnl_pct']).prod() - 1
                    win_r = (len(df_t[df_t['pnl_pct'] > 0]) / len(df_t)) * 100
                    status = "🟢 Long" if active_trade else "⚪ Waiting"
                    
                    results.append({
                        "Stock": sym,
                        "Returns %": round(ret * 100, 2),
                        "Win Rate %": round(win_r, 2),
                        "Total Trades": len(df_t),
                        "Current Status": status
                    })
        
        if results:
            res_df = pd.DataFrame(results).sort_values(by="Returns %", ascending=False)
            
            # Leaderboard Metrics
            c1, c2, c3 = st.columns(3)
            best = res_df.iloc[0]
            c1.metric("Top Performer", best['Stock'], f"{best['Returns %']}%")
            c2.metric("Avg Sector Return", f"{round(res_df['Returns %'].mean(), 2)}%")
            c3.metric("Stocks in Long", len(res_df[res_df['Current Status'] == "🟢 Long"]))

            st.divider()
            
            # Interactive Bar Chart
            
            fig = px.bar(res_df, x="Stock", y="Returns %", color="Returns %", 
                         color_continuous_scale='RdYlGn', title=f"Strategy Returns across {selected_sector}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Table
            st.subheader("Detailed Scan Report")
            st.dataframe(res_df, use_container_width=True)
        else:
            st.error("Could not fetch data for this sector. Please try again.")

else:
    st.info("Switch to 'Multi-Sector Scanner' in the sidebar to analyze entire NSE sectors at once.")