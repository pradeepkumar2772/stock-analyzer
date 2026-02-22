import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date, timedelta

# --- 1. SECTOR DEFINITIONS ---
SECTORS = {
    "Nifty Bank": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "AUBANK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "PNB.NS", "BANDHANBNK.NS"],
    "Nifty IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "LTIM.NS", "TECHM.NS", "PERSISTENT.NS", "MPHASIS.NS", "COFORGE.NS", "LTTS.NS"]
}

@dataclass
class Trade:
    symbol: str; direction: str; entry_date: datetime; entry_price: float
    exit_date: datetime = None; exit_price: float = None; exit_reason: str = None; pnl_pct: float = 0.0

# --- 2. BACKTEST ENGINE ---
def run_backtest(df, symbol, config, strategy_type):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()
    
    if strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)
    else: # EMA Ribbon
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = (df['ema_fast'] < df['ema_exit']) & (df['ema_fast'].shift(1) >= df['ema_exit'].shift(1))

    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            if sl_hit or tp_hit or prev['exit_signal']:
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = "SL" if sl_hit else ("TP" if tp_hit else "Signal")
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
    return trades

# --- 3. UI DASHBOARD ---
st.set_page_config(layout="wide", page_title="Sector Backtest Scanner")
st.sidebar.title("🎗️ Sector Scanner")
mode = st.sidebar.radio("Mode", ["Single Stock", "Sector Scanner"])
strat_choice = st.sidebar.selectbox("Strategy", ["RSI 60 Cross", "EMA Ribbon"])

config = {'use_sl': True, 'sl_val': 5.0, 'use_tp': True, 'tp_val': 25.0, 'use_slippage': True, 'slippage_val': 0.1}

if mode == "Single Stock":
    symbol = st.sidebar.text_input("Symbol", "BRITANNIA.NS").upper()
    if st.sidebar.button("🚀 Run Analysis"):
        # [Existing Phase 1 UI code for Single Stock goes here]
        pass

elif mode == "Sector Scanner":
    sector = st.sidebar.selectbox("Select Sector", list(SECTORS.keys()))
    scan_period = st.sidebar.selectbox("Scan Period", ["1 Year", "2 Years", "5 Years"])
    
    if st.sidebar.button("🔍 Start Sector Scan"):
        results = []
        progress_bar = st.progress(0)
        stock_list = SECTORS[sector]
        
        days = {"1 Year": 365, "2 Years": 730, "5 Years": 1825}[scan_period]
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        for idx, sym in enumerate(stock_list):
            progress_bar.progress((idx + 1) / len(stock_list))
            data = yf.download(sym, start=start_date, progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                data.columns = [str(c).lower() for c in data.columns]
                
                trades = run_backtest(data, sym, config, strat_choice)
                if trades:
                    df_t = pd.DataFrame([vars(t) for t in trades])
                    df_t['equity'] = 1000 * (1 + df_t['pnl_pct']).cumprod()
                    total_ret = (df_t['equity'].iloc[-1] / 1000 - 1) * 100
                    win_rate = (len(df_t[df_t['pnl_pct'] > 0]) / len(df_t)) * 100
                    results.append({"Stock": sym, "Returns (%)": round(total_ret, 2), "Win Rate (%)": round(win_rate, 2), "Trades": len(df_t)})
        
        if results:
            scan_df = pd.DataFrame(results).sort_values(by="Returns (%)", ascending=False)
            st.subheader(f"📊 {sector} Performance Leaderboard ({scan_period})")
            st.table(scan_df)
            
            
            
            fig = px.bar(scan_df, x="Stock", y="Returns (%)", color="Returns (%)", title=f"{sector} Strategy Returns Comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No trades generated for this sector in the given period.")