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

# --- 2. BACKTEST ENGINE (RSI 60 CROSS) ---
def run_backtest(df, symbol, config):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
    df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            
            if sl_hit or tp_hit or prev['exit_signal']:
                reason = "Stop Loss" if sl_hit else ("Target Profit" if tp_hit else "RSI Cross Down")
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade)
                active_trade = None
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
    return trades, df

# --- 3. UI & STYLING ---
st.set_page_config(layout="wide", page_title="Institutional Backtest Report")

st.markdown("""
    <style>
    .stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; }
    .report-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .report-table th { background-color: #2d2f3b; color: #bbb; padding: 12px; border: 1px solid #2d2f3b; font-size: 0.8rem; }
    .report-table td { border: 1px solid #2d2f3b; padding: 10px; text-align: center; color: #fff; font-size: 0.85rem; }
    .profit { background-color: #1b5e20 !important; color: #c8e6c9 !important; font-weight: bold; }
    .loss { background-color: #b71c1c !important; color: #ffcdd2 !important; font-weight: bold; }
    .total-cell { font-weight: bold; color: #fff; background-color: #1e3a5f !important; }
    .stat-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #2d2f3b; }
    .stat-label { color: #888; font-size: 0.9rem; }
    .stat-value { color: #fff; font-weight: bold; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🎗️ PK Ribbon Pro")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
tf_limits = {"Daily": "1d", "1 Hour": "1h", "15 Minutes": "15m", "5 Minutes": "5m"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_limits.keys()))
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2005-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

st.sidebar.divider()
st.sidebar.subheader("⚙️ Risk Management")
use_sl = st.sidebar.toggle("Enable Stop Loss", value=True)
sl_val = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0
use_tp = st.sidebar.toggle("Enable Target Profit", value=True)
tp_val = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0
use_slippage = st.sidebar.toggle("Apply Slippage", value=True)
slippage_val = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slippage else 0

if st.sidebar.button("🚀 Generate Institutional Report"):
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_limits[selected_tf], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            
            config = {'use_sl': use_sl, 'sl_val': sl_val, 'use_tp': use_tp, 'tp_val': tp_val, 'use_slippage': use_slippage, 'slippage_val': slippage_val, 'capital': capital}
            trades, processed_df = run_backtest(data, symbol, config)

            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
                df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                
                # Calculations
                wins = df_trades[df_trades['pnl_pct'] > 0]; losses = df_trades[df_trades['pnl_pct'] <= 0]
                total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                duration_delta = df_trades['exit_date'].max() - df_trades['entry_date'].min()
                years = max(duration_delta.days / 365.25, 0.1)
                cagr = (((df_trades['equity'].iloc[-1] / capital) ** (1/years)) - 1) * 100
                peak = df_trades['equity'].cummax(); drawdown = (df_trades['equity'] - peak) / peak; mdd = drawdown.min() * 100
                
                # Streak Logic
                pnl_bool = (df_trades['pnl_pct'] > 0).astype(int)
                streaks = pnl_bool.groupby((pnl_bool != pnl_bool.shift()).cumsum()).cumcount() + 1
                max_win_streak = streaks[pnl_bool == 1].max() if not wins.empty else 0
                max_loss_streak = streaks[pnl_bool == 0].max() if not losses.empty else 0

                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                
                with t1:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Returns (%)", f"{total_ret:.2f}%")
                    m2.metric("Max Drawdown(MDD)", f"{mdd:.2f}%")
                    m3.metric("Win Ratio", f"{(len(wins)/len(df_trades)*100):.2f}%")
                    m4.metric("Total Trades", len(df_trades))
                    st.divider()
                    st.subheader("Monthly Returns")
                    df_trades['year'] = df_trades['exit_date'].dt.year
                    df_trades['month'] = df_trades['exit_date'].dt.strftime('%b')
                    pivot = df_trades.groupby(['year', 'month'])['pnl_pct'].sum().unstack().fillna(0) * 100
                    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns])
                    pivot['Total'] = pivot.sum(axis=1)
                    html = "<table class='report-table'><thead><tr><th>Year</th>" + "".join([f"<th>{m}</th>" for m in pivot.columns]) + "</tr></thead><tbody>"
                    for year, row in pivot.iloc[::-1].iterrows():
                        html += f"<tr><td>{year}</td>"
                        for col_name, val in row.items():
                            cls = "profit" if val > 0 else ("loss" if val < 0 else "")
                            if col_name == "Total": cls = "total-cell"
                            display_val = f"{val:.2f}%" if val != 0 else "-"
                            html += f"<td class='{cls}'>{display_val}</td>"
                        html += "</tr>"
                    html += "</tbody></table>"
                    st.markdown(html, unsafe_allow_html=True)

                with t2:
                    st.subheader("Performance Characteristics")
                    col_l, col_r = st.columns([1, 2.5])
                    
                    with col_l:
                        with st.expander("📝 Backtest Details", expanded=True):
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Scrip</span><span class='stat-value'>{symbol}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Start Date</span><span class='stat-value'>{start_str}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>End Date</span><span class='stat-value'>{end_str}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Duration</span><span class='stat-value'>{duration_delta.days // 365} Years, {duration_delta.days % 365 // 30} Months</span></div>", unsafe_allow_html=True)
                        
                        with st.expander("📈 Return"):
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Total Return</span><span class='stat-value'>{total_ret:.2f} %</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>CAGR</span><span class='stat-value'>{cagr:.2f}%</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Avg Return/Trade</span><span class='stat-value'>{df_trades['pnl_pct'].mean()*100:.2f} %</span></div>", unsafe_allow_html=True)
                        
                        with st.expander("📉 Drawdown"):
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Maximum Drawdown</span><span class='stat-value'>{mdd:.2f} %</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Average Drawdown</span><span class='stat-value'>{drawdown.mean()*100:.2f} %</span></div>", unsafe_allow_html=True)

                        with st.expander("📊 Performance"):
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Win Rate</span><span class='stat-value'>{(len(wins)/len(df_trades)*100):.2f} %</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Avg Win Trade</span><span class='stat-value'>{wins['pnl_pct'].mean()*100:.2f} %</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Avg Loss Trade</span><span class='stat-value'>{losses['pnl_pct'].mean()*100:.2f} %</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Expectancy</span><span class='stat-value'>{(total_ret/len(df_trades)):.2f}</span></div>", unsafe_allow_html=True)

                        with st.expander("🔍 Trade Characteristics"):
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Total Trades</span><span class='stat-value'>{len(df_trades)}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Max Profit</span><span class='stat-value'>{df_trades['pnl_pct'].max()*100:.2f}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Max Loss</span><span class='stat-value'>{df_trades['pnl_pct'].min()*100:.2f}</span></div>", unsafe_allow_html=True)

                        with st.expander("🛡️ Risk-Adjusted Metrics"):
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Sharpe Ratio</span><span class='stat-value'>{(df_trades['pnl_pct'].mean()/df_trades['pnl_pct'].std()*np.sqrt(252)):.2f}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Calmar Ratio</span><span class='stat-value'>{abs(cagr/mdd):.2f}</span></div>", unsafe_allow_html=True)

                        with st.expander("⏱️ Holding Period"):
                            df_trades['hold_time'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.days
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Avg Holding Period</span><span class='stat-value'>{df_trades['hold_time'].mean():.1f} days</span></div>", unsafe_allow_html=True)
                        
                        with st.expander("🔥 Streak"):
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Win Streak</span><span class='stat-value'>{max_win_streak}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='stat-row'><span class='stat-label'>Loss Streak</span><span class='stat-value'>{max_loss_streak}</span></div>", unsafe_allow_html=True)
                            
                    with col_r:
                        st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Equity Curve", color_discrete_sequence=['#3498db']), use_container_width=True)
                        
                        st.plotly_chart(px.area(df_trades, x='exit_date', y=drawdown*100, title="Drawdown %", color_discrete_sequence=['#e74c3c']), use_container_width=True)

                with t3:
                    st.subheader("Distribution Analysis")
                    yearly_ret = df_trades.groupby('year')['pnl_pct'].sum() * 100
                    fig_p = go.Figure(data=[go.Bar(x=yearly_ret.index, y=yearly_ret.values, text=yearly_ret.values.round(1), texttemplate='%{text}%', textposition='outside', textfont=dict(color='white'), marker_color='#3498db')])
                    fig_p.add_hline(y=0, line_color="white")
                    st.plotly_chart(fig_p, use_container_width=True)

                with t4:
                    st.dataframe(df_trades[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl_pct', 'exit_reason']], use_container_width=True)
            else:
                st.warning("No trades found.")
    except Exception as e:
        st.error(f"Error: {e}")