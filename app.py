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

# --- 2. NIFTY 200 CONSTITUENTS ---
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

# --- 3. MULTI-STRATEGY ENGINE ---
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
                reason = "Stop Loss" if sl_hit else ("Target Profit" if tp_hit else "Signal")
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
    return trades, df

# --- 4. UI STYLING ---
st.set_page_config(layout="wide", page_title="Strategy Lab Pro")
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

# --- 5. SIDEBAR ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.selectbox("Symbol", NIFTY_200, index=NIFTY_200.index("BRITANNIA.NS"))
strat_choice = st.sidebar.selectbox("Select Strategy", ["RSI 60 Cross", "EMA Ribbon"])
tf_map = {"1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", "1 Hour": "1h", "Daily": "1d"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=4)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date(2010, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=date.today())

config = {}
if strat_choice == "EMA Ribbon":
    config['ema_fast'] = st.sidebar.number_input("Fast EMA", 20)
    config['ema_slow'] = st.sidebar.number_input("Slow EMA", 50)
    config['ema_exit'] = st.sidebar.number_input("Exit EMA", 30)

st.sidebar.divider()
use_sl = st.sidebar.toggle("Stop Loss", True); config['sl_val'] = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0; config['use_sl'] = use_sl
use_tp = st.sidebar.toggle("Target Profit", True); config['tp_val'] = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0; config['use_tp'] = use_tp
use_slip = st.sidebar.toggle("Slippage", True); config['slippage_val'] = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slip else 0; config['use_slippage'] = use_slip

# --- 6. EXECUTION ---
if st.sidebar.button("🚀 Run Backtest"):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval=tf_map[selected_tf], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            trades, processed_df = run_backtest(data, symbol, config, strat_choice)
            
            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
                df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                
                # Metric Calculations
                wins = df_trades[df_trades['pnl_pct'] > 0]; losses = df_trades[df_trades['pnl_pct'] <= 0]
                total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                duration = df_trades['exit_date'].max() - df_trades['entry_date'].min()
                years_v = max(duration.days / 365.25, 0.1)
                cagr = (((df_trades['equity'].iloc[-1] / capital) ** (1/years_v)) - 1) * 100
                peak = df_trades['equity'].cummax(); drawdown = (df_trades['equity'] - peak) / peak; mdd = drawdown.min() * 100
                sharpe = (df_trades['pnl_pct'].mean()/df_trades['pnl_pct'].std()*np.sqrt(252)) if len(df_trades)>1 else 0.0
                rr = (wins['pnl_pct'].mean()/abs(losses['pnl_pct'].mean())) if not losses.empty else 0.0
                exp = (total_ret/len(df_trades))
                calmar = abs(cagr/mdd) if mdd != 0 else 0.0
                pnl_b = (df_trades['pnl_pct'] > 0).astype(int); strk = pnl_b.groupby((pnl_b != pnl_b.shift()).cumsum()).cumcount() + 1
                max_w_s = strk[pnl_b == 1].max() if not wins.empty else 0; max_l_s = strk[pnl_b == 0].max() if not losses.empty else 0

                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                
                with t1:
                    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                    r1c1.metric("Total Returns (%)", f"{total_ret:.2f}%"); r1c2.metric("Max Drawdown(MDD)", f"{mdd:.2f}%"); r1c3.metric("Win Ratio", f"{(len(wins)/len(df_trades)*100):.2f}%"); r1c4.metric("Total Trades", len(df_trades))
                    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                    r2c1.metric("Initial Capital", f"{capital:,.2f}"); r2c2.metric("Final Capital", f"{df_trades['equity'].iloc[-1]:,.2f}"); r2c3.metric("CAGR", f"{cagr:.2f}%"); r2c4.metric("Avg Return/Trade", f"{(df_trades['pnl_pct'].mean()*100):.2f}%")
                    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
                    r3c1.metric("Risk-Reward Ratio", f"{rr:.2f}"); r3c2.metric("Expectancy", f"{exp:.2f}"); r3c3.metric("Sharpe Ratio", f"{sharpe:.2f}"); r3c4.metric("Calmer Ratio", f"{calmar:.2f}")
                    st.divider()
                    st.subheader("Monthly Returns")
                    df_trades['year'] = df_trades['exit_date'].dt.year; df_trades['month'] = df_trades['exit_date'].dt.strftime('%b')
                    piv = df_trades.groupby(['year', 'month'])['pnl_pct'].sum().unstack().fillna(0) * 100
                    piv = piv.reindex(columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']).fillna(0)
                    piv['Total'] = piv.sum(axis=1)
                    html = "<table class='report-table'><thead><tr><th>Year</th>" + "".join([f"<th>{m}</th>" for m in piv.columns]) + "</tr></thead><tbody>"
                    for yr, row in piv.iloc[::-1].iterrows():
                        html += f"<tr><td>{yr}</td>"
                        for col, val in row.items():
                            cls = "profit" if val > 0 else ("loss" if val < 0 else ""); cls = "total-cell" if col == "Total" else cls
                            html += f"<td class='{cls}'>{f'{val:.2f}%' if val != 0 else '-'}</td>"
                        html += "</tr>"
                    st.markdown(html + "</tbody></table>", unsafe_allow_html=True)
                    st.divider()
                    df_csv = pd.DataFrame({"Param": ["Total Return", "MDD", "Win Ratio", "CAGR", "Sharpe", "Expectancy"], "Value": [f"{total_ret:.2f}", f"{mdd:.2f}", f"{(len(wins)/len(df_trades)*100):.2f}", f"{cagr:.2f}", f"{sharpe:.2f}", f"{exp:.2f}"]})
                    st.download_button("📥 Download Quick Stats (CSV)", df_csv.to_csv(index=False).encode('utf-8'), f"{symbol}_Stats.csv", "text/csv")

                with t2:
                    cl, cr = st.columns([1, 2.5])
                    with cl:
                        with st.expander("📊 Backtest Details", expanded=True):
                            draw_stat("Strategy", strat_choice); draw_stat("Scrip", symbol); draw_stat("Period", f"{start_date} to {end_date}"); draw_stat("Duration", f"{duration.days // 365}Y, {duration.days % 365 // 30}M")
                        with st.expander("📈 Return"):
                            draw_stat("Total Return", f"{total_ret:.2f} %"); draw_stat("CAGR", f"{cagr:.2f}%"); draw_stat("Avg Return Trade", f"{df_trades['pnl_pct'].mean()*100:.2f} %"); draw_stat("Highest Return", f"{df_trades['pnl_pct'].max()*100:.2f} %"); draw_stat("Lowest Return", f"{df_trades['pnl_pct'].min()*100:.2f} %")
                        with st.expander("📉 Drawdown"):
                            draw_stat("Maximum Drawdown", f"{mdd:.2f} %"); draw_stat("Average Drawdown", f"{drawdown.mean()*100:.2f} %")
                        with st.expander("🏆 Performance"):
                            draw_stat("Win Rate", f"{(len(wins)/len(df_trades)*100):.2f} %"); draw_stat("Loss Rate", f"{(len(losses)/len(df_trades)*100):.2f} %"); draw_stat("Avg Return/Win", f"{wins['pnl_pct'].mean()*100:.2f} %"); draw_stat("Avg Return/Loss", f"{losses['pnl_pct'].mean()*100:.2f} %"); draw_stat("Risk Reward Ratio", f"{rr:.2f}"); draw_stat("Expectancy", f"{exp:.2f}")
                        with st.expander("🔍 Trade Characteristics"):
                            draw_stat("Total Number of Trades", len(df_trades)); draw_stat("Profit Trades", len(wins)); draw_stat("Loss Trades", len(losses)); draw_stat("Max Profit", f"{df_trades['pnl_pct'].max()*100:.2f}"); draw_stat("Max Loss", f"{df_trades['pnl_pct'].min()*100:.2f}"); draw_stat("Winning Streak", f"{max_w_s}.00"); draw_stat("Lossing Streak", f"{max_l_s}.00")
                        with st.expander("🛡️ Risk-Adjusted Metrics"):
                            draw_stat("Sharpe Ratio", f"{sharpe:.2f}"); draw_stat("Calmar Ratio", f"{calmar:.2f}")
                        with st.expander("⏱️ Holding Period"):
                            df_trades['hold'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.days
                            draw_stat("Max Hold", f"{df_trades['hold'].max()} days"); draw_stat("Min Hold", f"{df_trades['hold'].min()} days"); draw_stat("Avg Hold", f"{df_trades['hold'].mean():.2f} days")
                        with st.expander("🔥 Streak"):
                            draw_stat("Win Streak", max_w_s); draw_stat("Loss Streak", max_l_s)
                    with cr:
                        st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Equity Curve", color_discrete_sequence=['#3498db']), use_container_width=True)
                        st.plotly_chart(px.area(df_trades, x='exit_date', y=drawdown*100, title="Underwater Drawdown (%)", color_discrete_sequence=['#e74c3c']), use_container_width=True)

                with t3:
                    y_r = df_trades.groupby('year')['pnl_pct'].sum() * 100
                    st.plotly_chart(go.Figure(data=[go.Bar(x=y_r.index, y=y_r.values, text=y_r.values.round(1), texttemplate='%{text}%', textposition='outside', textfont=dict(color='white'), marker=dict(color='#3498db'))]).add_hline(y=0, line_color="white").update_layout(title="1. Return by Period (%)", template="plotly_dark"), use_container_width=True)
                    wl_y = df_trades.assign(is_win=df_trades['pnl_pct'] > 0).groupby(['year', 'is_win']).size().unstack(fill_value=0)
                    wl_y.columns = ['Losers', 'Winners']; wl_y['Total'] = wl_y['Winners'] + wl_y['Losers']
                    f2 = go.Figure()
                    for c, color in zip(['Total', 'Winners', 'Losers'], ['#3498db', '#2ecc71', '#e74c3c']): f2.add_trace(go.Bar(x=wl_y.index, y=wl_y[c], name=c, marker=dict(color=color), text=wl_y[c], textposition='outside', textfont=dict(color='white')))
                    st.plotly_chart(f2.update_layout(barmode='group', title="2. Winners/Losers (Yearly)", template="plotly_dark"), use_container_width=True)
                    ex_st = df_trades['exit_reason'].value_counts(normalize=True) * 100
                    st.plotly_chart(go.Figure(data=[go.Bar(x=ex_st.index, y=ex_st.values, text=ex_st.values.astype(int), texttemplate='%{text}%', textposition='outside', textfont=dict(color='white'), marker=dict(color='#3498db'))]).update_layout(title="3. Exits Distribution", template="plotly_dark"), use_container_width=True)
                    df_trades['day'] = df_trades['exit_date'].dt.day_name(); day_d = df_trades.assign(is_win=df_trades['pnl_pct'] > 0).groupby(['day', 'is_win']).size().unstack(fill_value=0)
                    day_d.columns = ['Losers', 'Winners']; day_d['Total'] = day_d['Winners'] + day_d['Losers']; day_d = day_d.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
                    f4 = go.Figure()
                    for c, color in zip(['Total', 'Winners', 'Losers'], ['#3498db', '#2ecc71', '#e74c3c']): f4.add_trace(go.Bar(x=day_d.index, y=day_d[c], name=c, marker=dict(color=color), text=day_d[c], textposition='outside', textfont=dict(color='white')))
                    st.plotly_chart(f4.update_layout(barmode='group', title="4. Trades by Day of the Week", template="plotly_dark"), use_container_width=True)

                with t4:
                    st.dataframe(df_trades[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl_pct', 'exit_reason']], use_container_width=True)
            else:
                st.warning("No trades found for the selected parameters and date range.")
    except Exception as e:
        st.error(f"Execution Error: {e}")