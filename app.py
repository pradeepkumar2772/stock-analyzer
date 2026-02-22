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

# --- RS-AV CALCULATION FUNCTION ---
def calculate_rsav(stock_df, benchmark_df, lookback=50):
    s_roc = stock_df['close'].pct_change(lookback) * 100
    s_vol = s_roc.rolling(window=lookback).std()
    s_net = s_roc - s_vol
    b_roc = benchmark_df['close'].pct_change(lookback) * 100
    b_vol = b_roc.rolling(window=lookback).std()
    b_net = b_roc - b_vol
    return s_net - b_net

# --- 2. MULTI-STRATEGY ENGINE ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    if config.get('use_rsav', False) and benchmark_df is not None:
        df['rsav'] = calculate_rsav(df, benchmark_df, lookback=50)

    # Indicator Calculations
    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Base Indicators
    for span in [config.get('ema_fast', 20), config.get('ema_slow', 50), config.get('ema_exit', 30)]:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    # ATR
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    # --- Strategy Signals ---
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20'].shift(1)) & (df['close'] > df['ema_20'])
        df['exit_signal'] = (df['close'] < df['ema_15'])
    elif strategy_type == "EMA Ribbon":
        f, s, e = config['ema_fast'], config['ema_slow'], config['ema_exit']
        df['long_signal'] = (df[f'ema_{f}'] > df[f'ema_{s}']) & (df[f'ema_{f}'].shift(1) <= df[f'ema_{s}'].shift(1))
        df['exit_signal'] = (df[f'ema_{f}'] < df[f'ema_{e}'])

    # --- Backtest Loop ---
    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        market_ok = True if not config.get('use_rsav', False) else current['rsav'] >= config.get('rsav_trigger', -0.5)
        
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            if sl_hit or prev['exit_signal']:
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = "SL" if sl_hit else "Signal"
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        elif prev['long_signal'] and market_ok:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
    return trades, df

# --- 3. UI STYLING ---
st.set_page_config(layout="wide", page_title="Pro-Tracer Pro")
st.markdown("<style>.stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; } .report-table { width: 100%; border-collapse: collapse; margin-top: 10px; } .report-table td { border: 1px solid #2d2f3b; padding: 10px; text-align: center; color: #fff; font-size: 0.85rem; } .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; } .stat-label { color: #999; font-size: 0.85rem; } .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }</style>", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strat_choice = st.sidebar.selectbox("Select Strategy", ["PK Strategy (Positional)", "EMA Ribbon"])
selected_tf = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=4)
capital = st.sidebar.number_input("Initial Capital", value=100000.0)
start_str = st.sidebar.text_input("Start Date", value="1999-01-01")

st.sidebar.divider()
use_rsav = st.sidebar.toggle("Enable RS-AV Filter", False)
rsav_trig = st.sidebar.number_input("RS-AV Trigger", value=-0.5, step=0.1)

config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 15, 'use_rsav': use_rsav, 'rsav_trigger': rsav_trig, 'use_sl': True, 'sl_val': 5.0, 'use_tp': False, 'use_slippage': True, 'slippage_val': 0.1}

# --- 5. EXECUTION ---
if st.sidebar.button("🚀 Run Backtest"):
    try:
        data = yf.download(symbol, start=start_str, auto_adjust=True)
        bench_data = yf.download("^NSEI", start=start_str, auto_adjust=True) if use_rsav else None
        
        if not data.empty:
            for d in [data, bench_data]:
                if d is not None:
                    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                    d.columns = [str(col).lower() for col in d.columns]
            
            if use_rsav and bench_data is not None:
                common_idx = data.index.intersection(bench_data.index)
                data, bench_data = data.loc[common_idx], bench_data.loc[common_idx]

            trades, processed_df = run_backtest(data, symbol, config, strat_choice, bench_data)
            
            if trades:
                df_t = pd.DataFrame([vars(t) for t in trades])
                df_t['entry_date'] = pd.to_datetime(df_t['entry_date']); df_t['exit_date'] = pd.to_datetime(df_t['exit_date'])
                df_t['equity'] = capital * (1 + df_t['pnl_pct']).cumprod()
                
                # Calculations for all 8 expanders
                wins = df_t[df_t['pnl_pct'] > 0]; losses = df_t[df_t['pnl_pct'] <= 0]
                total_ret = (df_t['equity'].iloc[-1] / capital - 1) * 100
                duration = df_t['exit_date'].max() - df_t['entry_date'].min()
                years = max(duration.days / 365.25, 0.1)
                cagr = (((df_t['equity'].iloc[-1] / capital) ** (1/years)) - 1) * 100
                drawdown = (df_t['equity'] - df_t['equity'].cummax()) / df_t['equity'].cummax()
                df_t['hold'] = (df_t['exit_date'] - df_t['entry_date']).dt.days
                
                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                
                with t2:
                    cl, cr = st.columns([1, 2.5])
                    with cl:
                        with st.expander("📝 Backtest Details", expanded=True):
                            draw_stat("Scrip", symbol); draw_stat("Start Date", df_t['entry_date'].min().strftime('%d-%b %y'))
                            draw_stat("End Date", df_t['exit_date'].max().strftime('%d-%b %y'))
                            draw_stat("Duration", f"{duration.days // 365} Years, {(duration.days % 365) // 30} Months")
                            draw_stat("Segment", "NSE"); draw_stat("Timeframe", selected_tf)
                        
                        with st.expander("💰 Return"):
                            draw_stat("Total Return", f"{total_ret:.2f} %"); draw_stat("CAGR", f"{cagr:.2f}%")
                            draw_stat("Avg Return Per Trade", f"{df_t['pnl_pct'].mean()*100:.2f} %")
                            draw_stat("Highest Return", f"{df_t['pnl_pct'].max()*100:.2f} %"); draw_stat("Lowest Return", f"{df_t['pnl_pct'].min()*100:.2f} %")
                        
                        with st.expander("📉 Drawdown"):
                            draw_stat("Maximum Drawdown", f"{drawdown.min()*100:.2f} %"); draw_stat("Average Drawdown", f"{drawdown.mean()*100:.2f} %")
                        
                        with st.expander("🏆 Performance"):
                            draw_stat("Win Rate", f"{(len(wins)/len(df_t)*100):.2f} %"); draw_stat("Loss Rate", f"{(len(losses)/len(df_t)*100):.2f} %")
                            draw_stat("Avg Win Trade", f"{wins['pnl_pct'].mean()*100:.2f} %"); draw_stat("Avg Loss Trade", f"{losses['pnl_pct'].mean()*100:.2f} %")
                            draw_stat("Risk Reward Ratio", f"{abs(wins['pnl_pct'].mean()/losses['pnl_pct'].mean()):.2f}" if not losses.empty else "0.0")
                            draw_stat("Expectancy", f"{(total_ret/len(df_t)):.2f}")

                        with st.expander("🔍 Trade Characteristics"):
                            draw_stat("Total Trades", len(df_t)); draw_stat("Profit Trades", len(wins)); draw_stat("Loss Trades", len(losses))
                            draw_stat("Max Profit", f"{df_t['pnl_pct'].max()*100:.2f}"); draw_stat("Max Loss", f"{df_t['pnl_pct'].min()*100:.2f}")

                        with st.expander("🛡️ Risk-Adjusted Metrics"):
                            sharpe = (df_t['pnl_pct'].mean()/df_t['pnl_pct'].std()*np.sqrt(252)) if len(df_t)>1 else 0.0
                            draw_stat("Sharpe Ratio", f"{sharpe:.2f}"); draw_stat("Calmar Ratio", f"{abs(cagr/(drawdown.min()*100)):.2f}" if drawdown.min() !=0 else "0.0")

                        with st.expander("⏱️ Holding Period"):
                            draw_stat("Max Holding Period", f"{df_t['hold'].max()} days"); draw_stat("Min Holding Period", f"{df_t['hold'].min()} days"); draw_stat("Average Holding Period", f"{df_t['hold'].mean():.2f} days")

                        with st.expander("🔥 Streak"):
                            pnl_b = (df_t['pnl_pct'] > 0).astype(int); strk = pnl_b.groupby((pnl_b != pnl_b.shift()).cumsum()).cumcount() + 1
                            draw_stat("Win Streak", strk[pnl_b == 1].max() if not wins.empty else 0); draw_stat("Loss Streak", strk[pnl_b == 0].max() if not losses.empty else 0)
                    with cr:
                        st.plotly_chart(px.line(df_t, x='exit_date', y='equity', title="Capital Growth"), use_container_width=True)
            else: st.warning("No trades found.")
    except Exception as e: st.error(f"Error: {e}")