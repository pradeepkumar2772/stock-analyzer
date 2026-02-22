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

# --- 2. RSI 60 CROSS ENGINE ---
def run_backtest(df, symbol, config):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    # RSI Calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    # Strategy: Entry on RSI Cross 60 Up | Exit on RSI Cross 60 Down
    df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
    df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            
            if sl_hit or tp_hit or prev['exit_signal']:
                reason = "Stop Loss" if sl_hit else ("Target Profit" if tp_hit else "System Builder")
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
st.set_page_config(layout="wide", page_title="RSI Pro Report")

st.markdown("""
    <style>
    .stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; }
    .report-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .report-table td { border: 1px solid #2d2f3b; padding: 10px; text-align: center; color: #fff; font-size: 0.85rem; }
    .profit { background-color: #1b5e20 !important; color: #c8e6c9 !important; font-weight: bold; }
    .loss { background-color: #b71c1c !important; color: #ffcdd2 !important; font-weight: bold; }
    .total-cell { font-weight: bold; color: #fff; background-color: #1e3a5f !important; }
    .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; }
    .stat-label { color: #999; font-size: 0.85rem; }
    .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("🎗️ RSI 60 Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
selected_tf = st.sidebar.selectbox("Timeframe", ["Daily", "1 Hour", "15 Minutes"])
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2005-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))
st.sidebar.divider()
use_sl = st.sidebar.toggle("Stop Loss", value=True); sl_val = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0
use_tp = st.sidebar.toggle("Target Profit", value=True); tp_val = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0
use_slippage = st.sidebar.toggle("Slippage", value=True); slippage_val = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slippage else 0

if st.sidebar.button("🚀 Run Backtest"):
    try:
        tf_map = {"Daily": "1d", "1 Hour": "1h", "15 Minutes": "15m"}
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            
            trades, processed_df = run_backtest(data, symbol, {'use_sl': use_sl, 'sl_val': sl_val, 'use_tp': use_tp, 'tp_val': tp_val, 'use_slippage': use_slippage, 'slippage_val': slippage_val})

            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
                df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                
                # METRICS CALCULATIONS
                wins = df_trades[df_trades['pnl_pct'] > 0]; losses = df_trades[df_trades['pnl_pct'] <= 0]
                total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                duration = df_trades['exit_date'].max() - df_trades['entry_date'].min()
                years_val = max(duration.days / 365.25, 0.1)
                cagr = (((df_trades['equity'].iloc[-1] / capital) ** (1/years_val)) - 1) * 100
                peak = df_trades['equity'].cummax(); drawdown = (df_trades['equity'] - peak) / peak; mdd = drawdown.min() * 100
                
                pnl_bool = (df_trades['pnl_pct'] > 0).astype(int)
                streaks = pnl_bool.groupby((pnl_bool != pnl_bool.shift()).cumsum()).cumcount() + 1
                max_w_streak = streaks[pnl_bool == 1].max() if not wins.empty else 0
                max_l_streak = streaks[pnl_bool == 0].max() if not losses.empty else 0

                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                
                with t1:
                    # FULL 12 METRIC GRID
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Returns (%)", f"{total_ret:.2f}%"); c2.metric("Max Drawdown(MDD)", f"{mdd:.2f}%"); c3.metric("Win Ratio", f"{(len(wins)/len(df_trades)*100):.2f}%"); c4.metric("Total Trades", len(df_trades))
                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("Initial Capital", f"{capital:,.2f}"); c6.metric("Final Capital", f"{df_trades['equity'].iloc[-1]:,.2f}"); c7.metric("CAGR", f"{cagr:.2f}%"); c8.metric("Avg Return/Trade", f"{(df_trades['pnl_pct'].mean()*100):.2f}%")
                    c9, c10, c11, c12 = st.columns(4)
                    c9.metric("Risk-Reward Ratio", f"{(wins['pnl_pct'].mean()/abs(losses['pnl_pct'].mean())):.2f}" if not losses.empty else "N/A"); c10.metric("Expectancy", f"{(total_ret/len(df_trades)):.2f}"); c11.metric("Sharpe Ratio", f"{(df_trades['pnl_pct'].mean()/df_trades['pnl_pct'].std()*np.sqrt(252)):.2f}" if len(df_trades)>1 else "0.0"); c12.metric("Calmer Ratio", f"{abs(cagr/mdd):.2f}" if mdd != 0 else "N/A")
                    st.divider()
                    st.subheader("Monthly Returns")
                    df_trades['year'] = df_trades['exit_date'].dt.year; df_trades['month'] = df_trades['exit_date'].dt.strftime('%b')
                    pivot = df_trades.groupby(['year', 'month'])['pnl_pct'].sum().unstack().fillna(0) * 100
                    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns]); pivot['Total'] = pivot.sum(axis=1)
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
                    cl, cr = st.columns([1, 2.5])
                    with cl:
                        with st.expander("📊 Backtest Details", expanded=True):
                            draw_stat("Scrip", symbol.split('.')[0]); draw_stat("Start Date", start_str); draw_stat("End Date", end_str); draw_stat("Duration", f"{duration.days // 365}Y, {duration.days % 365 // 30}M"); draw_stat("Segment", "NSE"); draw_stat("Timeframe", selected_tf)
                        with st.expander("📈 Return"):
                            draw_stat("Total Return", f"{total_ret:.2f} %"); draw_stat("CAGR", f"{cagr:.2f}%"); draw_stat("Avg Return Trade", f"{df_trades['pnl_pct'].mean()*100:.2f} %"); draw_stat("Highest Return", f"{df_trades['pnl_pct'].max()*100:.2f} %"); draw_stat("Lowest Return", f"{df_trades['pnl_pct'].min()*100:.2f} %")
                        with st.expander("📉 Drawdown"):
                            draw_stat("Max Drawdown", f"{mdd:.2f} %"); draw_stat("Avg Drawdown", f"{drawdown.mean()*100:.2f} %")
                        with st.expander("🏆 Performance"):
                            draw_stat("Win Rate", f"{(len(wins)/len(df_trades)*100):.2f} %"); draw_stat("Loss Rate", f"{(len(losses)/len(df_trades)*100):.2f} %"); draw_stat("Avg Win Trade", f"{wins['pnl_pct'].mean()*100:.2f} %"); draw_stat("Avg Loss Trade", f"{losses['pnl_pct'].mean()*100:.2f} %"); draw_stat("Risk Reward Ratio", f"{(wins['pnl_pct'].mean()/abs(losses['pnl_pct'].mean())):.2f}" if not losses.empty else "N/A"); draw_stat("Expectancy", f"{(total_ret/len(df_trades)):.2f}")
                        with st.expander("🔍 Trade Characteristics"):
                            draw_stat("Total Trades", len(df_trades)); draw_stat("Profit Trades", len(wins)); draw_stat("Loss Trades", len(losses)); draw_stat("Max Profit", f"{df_trades['pnl_pct'].max()*100:.2f}"); draw_stat("Max Loss", f"{df_trades['pnl_pct'].min()*100:.2f}"); draw_stat("Win Streak", f"{max_w_streak}.00"); draw_stat("Loss Streak", f"{max_l_streak}.00")
                        with st.expander("🛡️ Risk-Adjusted Metrics"):
                            draw_stat("Sharpe Ratio", f"{(df_trades['pnl_pct'].mean()/df_trades['pnl_pct'].std()*np.sqrt(252)):.2f}" if len(df_trades)>1 else "0.0"); draw_stat("Calmar Ratio", f"{abs(cagr/mdd):.2f}" if mdd != 0 else "N/A")
                        with st.expander("⏱️ Holding Period"):
                            df_trades['hold'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.days
                            draw_stat("Max Hold", f"{df_trades['hold'].max()} days"); draw_stat("Min Hold", f"{df_trades['hold'].min()} days"); draw_stat("Avg Hold", f"{df_trades['hold'].mean():.2f} days")
                        with st.expander("🔥 Streak"):
                            draw_stat("Win Streak", max_w_streak); draw_stat("Loss Streak", max_l_streak)
                    with cr:
                        st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Equity Curve Strategy", color_discrete_sequence=['#3498db']), use_container_width=True)
                        
                        st.plotly_chart(px.area(df_trades, x='exit_date', y=drawdown*100, title="Underwater Drawdown (%)", color_discrete_sequence=['#e74c3c']), use_container_width=True)

                with t3:
                    # 1. RETURN BY PERIOD
                    y_ret = df_trades.groupby('year')['pnl_pct'].sum() * 100
                    fig1 = go.Figure(data=[go.Bar(x=y_ret.index, y=y_ret.values, text=y_ret.values.round(1), texttemplate='%{text}%', textposition='outside', textfont=dict(color='white'), marker_color='#3498db')])
                    fig1.add_hline(y=0, line_color="white"); st.plotly_chart(fig1, use_container_width=True)
                    # 2. WINNERS/LOSERS
                    wl_y = df_trades.assign(is_win=df_trades['pnl_pct'] > 0).groupby(['year', 'is_win']).size().unstack(fill_value=0)
                    wl_y.columns = ['Losers', 'Winners']; wl_y['Total'] = wl_y['Winners'] + wl_y['Losers']
                    fig2 = go.Figure()
                    for c, color in zip(['Total', 'Winners', 'Losers'], ['#3498db', '#2ecc71', '#e74c3c']):
                        fig2.add_trace(go.Bar(x=wl_y.index, y=wl_y[c], name=c, marker_color=color, text=wl_y[c], textposition='outside', textfont=dict(color='white')))
                    st.plotly_chart(fig2, use_container_width=True)
                    # 3. EXITS DISTRIBUTION
                    ex_st = df_trades['exit_reason'].value_counts(normalize=True) * 100
                    fig3 = go.Figure(data=[go.Bar(x=ex_st.index, y=ex_st.values, text=ex_st.values.astype(int), texttemplate='%{text}%', textposition='outside', textfont=dict(color='white'), marker_color='#3498db')])
                    st.plotly_chart(fig3, use_container_width=True)
                    # 4. TRADES BY DAY
                    df_trades['day'] = df_trades['exit_date'].dt.day_name(); day_d = df_trades.assign(is_win=df_trades['pnl_pct'] > 0).groupby(['day', 'is_win']).size().unstack(fill_value=0)
                    day_d.columns = ['Losers', 'Winners']; day_d['Total'] = day_d['Winners'] + day_d['Losers']; day_d = day_d.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
                    fig4 = go.Figure()
                    for c, color in zip(['Total', 'Winners', 'Losers'], ['#3498db', '#2ecc71', '#e74c3c']):
                        fig4.add_trace(go.Bar(x=day_d.index, y=day_d[c], name=c, marker_color=color, text=day_d[c], textposition='outside', textfont=dict(color='white')))
                    st.plotly_chart(fig4, use_container_width=True)

                with t4:
                    st.dataframe(df_trades[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl_pct', 'exit_reason']], use_container_width=True)
    except Exception as e:
        st.error(f"Execution Error: {e}")