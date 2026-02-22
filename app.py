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

# --- 2. MULTI-STRATEGY ENGINE ---
def run_backtest(df, symbol, config, strategy_type):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    # --- Indicator Math ---
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    # --- Candlestick Definitions (Nison Principles) ---
    body_size = np.abs(df['close'] - df['open'])
    lower_shadow = np.minimum(df['open'], df['close']) - df['low']
    upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
    candle_range = df['high'] - df['low']
    is_green = df['close'] > df['open']
    is_red = df['close'] < df['open']

    # 1. Hammer
    df['is_hammer'] = (lower_shadow > (2 * body_size)) & (upper_shadow < (0.1 * candle_range))
    
    # 2. Bullish Engulfing
    df['is_bull_engulf'] = is_green & is_red.shift(1) & \
                           (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))

    # 3. Piercing Line (Fixed the mid_point variable)
    mid_point_prev = (df['open'].shift(1) + df['close'].shift(1)) / 2
    df['is_piercing'] = is_red.shift(1) & is_green & \
                        (df['open'] < df['low'].shift(1)) & (df['close'] > mid_point_prev)

    # 4. Morning Star (Three Candle Bottom Reversal)
    # Candle 1: Long Red, Candle 2: Small Body Star (Gap Down), Candle 3: Long Green
    is_star = body_size < (body_size.rolling(window=10).mean() * 0.5)
    df['is_morning_star'] = is_red.shift(2) & is_star.shift(1) & is_green & \
                            (df['open'].shift(1) < df['close'].shift(2)) & \
                            (df['close'] > mid_point_prev.shift(1))

    # --- Strategy Signals ---
    if strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)

    elif strategy_type == "EMA Ribbon":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = (df['ema_fast'] < df['ema_exit']) & (df['ema_fast'].shift(1) >= df['ema_exit'].shift(1))

    elif strategy_type == "Nison: Morning Star":
        df['long_signal'] = df['is_morning_star']
        df['exit_signal'] = (df['close'] < df['low'].rolling(window=3).min()) | (df['close'] < df['ema_exit'])

    elif strategy_type == "Nison: Hammer Reversal":
        df['long_signal'] = df['is_hammer'] & (df['close'] < df['ema_fast'])
        df['exit_signal'] = (df['close'] > df['ema_fast'] * 1.05) | (df['close'] < df['low'].shift(1))

    elif strategy_type == "Nison: Bullish Engulfing":
        df['long_signal'] = df['is_bull_engulf'] & (df['close'] > df['sma_200'])
        df['exit_signal'] = (df['close'] < df['ema_exit'])

    elif strategy_type == "Nison: Piercing Line":
        df['long_signal'] = df['is_piercing']
        df['exit_signal'] = (df['close'] < df['ema_exit'])

    elif strategy_type == "EMA & RSI Synergy":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['rsi'] > 60)
        df['exit_signal'] = (df['close'] < df['ema_exit']) | (df['rsi'] < 40)

    elif strategy_type == "Double Bottom Breakout":
        neckline = df['high'].rolling(window=20).max()
        df['long_signal'] = (df['close'] > neckline.shift(1))
        df['exit_signal'] = (df['close'] < df['ema_exit'])

    elif strategy_type == "Fibonacci 61.8% Retracement":
        hhv = df['high'].rolling(window=20).max()
        llv = df['low'].rolling(window=20).min()
        fib = hhv - ((hhv - llv) * 0.618)
        df['long_signal'] = (df['close'] > df['sma_200']) & (df['low'] <= fib) & (df['close'] > df['high'].shift(1))
        df['exit_signal'] = df['close'] < llv.shift(1)

    elif strategy_type == "Relative Strength Play":
        stock_ret = df['close'].pct_change(periods=55)
        df['long_signal'] = (stock_ret > 0) & (df['close'] > df['ema_fast'])
        df['exit_signal'] = (df['close'] < df['ema_slow'])

    elif strategy_type == "ATR Band Breakout":
        sma_20 = df['close'].rolling(window=20).mean()
        df['atr'] = (df['high'] - df['low']).rolling(window=14).mean()
        u_atr = sma_20 + df['atr']; l_atr = sma_20 - df['atr']
        df['long_signal'] = (df['close'] > u_atr) & (df['close'].shift(1) <= u_atr.shift(1))
        df['exit_signal'] = (df['close'] < l_atr) & (df['close'].shift(1) >= l_atr.shift(1))

    # --- Backtest Loop ---
    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        if active_trade:
            sl_hit = config['use_sl'] and current['low'] <= active_trade.entry_price * (1 - config['sl_val'] / 100)
            tp_hit = config['use_tp'] and current['high'] >= active_trade.entry_price * (1 + config['tp_val'] / 100)
            if sl_hit or tp_hit or prev['exit_signal']:
                reason = "Stop Loss" if sl_hit else ("Target Profit" if tp_hit else "Signal Exit")
                active_trade.exit_price = current['open'] * (1 - slippage)
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                trades.append(active_trade); active_trade = None
        elif prev['long_signal']:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
    return trades, df

# --- 3. UI STYLING & SIDEBAR ---
st.set_page_config(layout="wide", page_title="Strategy Lab Pro")
st.markdown("<style>.stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; } .report-table { width: 100%; border-collapse: collapse; margin-top: 10px; } .report-table td { border: 1px solid #2d2f3b; padding: 10px; text-align: center; color: #fff; font-size: 0.85rem; } .profit { background-color: #1b5e20 !important; color: #c8e6c9 !important; font-weight: bold; } .loss { background-color: #b71c1c !important; color: #ffcdd2 !important; font-weight: bold; } .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; } .stat-label { color: #999; font-size: 0.85rem; } .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }</style>", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strat_choice = st.sidebar.selectbox("Select Strategy", ["RSI 60 Cross", "EMA Ribbon", "Nison: Morning Star", "Nison: Hammer Reversal", "Nison: Bullish Engulfing", "Nison: Piercing Line", "EMA & RSI Synergy", "Double Bottom Breakout", "Fibonacci 61.8% Retracement", "Relative Strength Play", "ATR Band Breakout"])
tf_map = {"1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", "1 Hour": "1h", "Daily": "1d"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=4)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2005-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 30, 'hhv_period': 20}
st.sidebar.divider()
use_sl = st.sidebar.toggle("Stop Loss", True); config['sl_val'] = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0; config['use_sl'] = use_sl
use_tp = st.sidebar.toggle("Target Profit", True); config['tp_val'] = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0; config['use_tp'] = use_tp
use_slip = st.sidebar.toggle("Slippage", True); config['slippage_val'] = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slip else 0; config['use_slippage'] = use_slip

# --- 4. EXECUTION ---
if st.sidebar.button("🚀 Run Backtest"):
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            trades, processed_df = run_backtest(data, symbol, config, strat_choice)
            
            if trades:
                df_trades = pd.DataFrame([vars(t) for t in trades])
                df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date']); df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
                df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()
                df_trades['year'] = df_trades['exit_date'].dt.year; df_trades['month'] = df_trades['exit_date'].dt.strftime('%b')
                
                wins = df_trades[df_trades['pnl_pct'] > 0]; losses = df_trades[df_trades['pnl_pct'] <= 0]
                total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
                duration = df_trades['exit_date'].max() - df_trades['entry_date'].min()
                years_v = max(duration.days / 365.25, 0.1)
                cagr = (((df_trades['equity'].iloc[-1] / capital) ** (1/years_v)) - 1) * 100
                peak = df_trades['equity'].cummax(); drawdown = (df_trades['equity'] - peak) / peak; mdd = drawdown.min() * 100
                sharpe = (df_trades['pnl_pct'].mean()/df_trades['pnl_pct'].std()*np.sqrt(252)) if len(df_trades)>1 else 0.0
                rr = (wins['pnl_pct'].mean()/abs(losses['pnl_pct'].mean())) if not losses.empty else 0.0
                exp = (total_ret/len(df_trades)); calmar = abs(cagr/mdd) if mdd != 0 else 0.0
                pnl_b = (df_trades['pnl_pct'] > 0).astype(int); strk = pnl_b.groupby((pnl_b != pnl_b.shift()).cumsum()).cumcount() + 1
                max_w_s = strk[pnl_b == 1].max() if not wins.empty else 0; max_l_s = strk[pnl_b == 0].max() if not losses.empty else 0

                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                with t1:
                    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                    r1c1.metric("Total Returns (%)", f"{total_ret:.2f}%"); r1c2.metric("Max Drawdown(MDD)", f"{mdd:.2f}%"); r1c3.metric("Win Ratio", f"{(len(wins)/len(df_trades)*100):.2f}%"); r1c4.metric("Total Trades", len(df_trades))
                    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                    r2c1.metric("Initial Capital", f"{capital:,.2f}"); r2c2.metric("Final Capital", f"{df_trades['equity'].iloc[-1]:,.2f}"); r2c3.metric("CAGR", f"{cagr:.2f}%"); r2c4.metric("Avg Return/Trade", f"{(df_trades['pnl_pct'].mean()*100):.2f}%")
                    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
                    r3c1.metric("Risk-Reward Ratio", f"{rr:.2f}"); r3c2.metric("Expectancy", f"{exp:.2f}"); r3c3.metric("Sharpe Ratio", f"{sharpe:.2f}"); r3c4.metric("Calmar Ratio", f"{calmar:.2f}")
                    st.divider(); st.subheader("Monthly Returns")
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

                with t2:
                    cl, cr = st.columns([1, 2.5])
                    with cl:
                        with st.expander("📈 Return"):
                            draw_stat("Total Return", f"{total_ret:.2f} %"); draw_stat("CAGR", f"{cagr:.2f}%"); draw_stat("Avg Return Trade", f"{df_trades['pnl_pct'].mean()*100:.2f} %")
                        with st.expander("📉 Drawdown"):
                            draw_stat("Maximum Drawdown", f"{mdd:.2f} %"); draw_stat("Average Drawdown", f"{drawdown.mean()*100:.2f} %")
                        with st.expander("🏆 Performance"):
                            draw_stat("Win Rate", f"{(len(wins)/len(df_trades)*100):.2f} %"); draw_stat("Risk Reward Ratio", f"{rr:.2f}")
                        with st.expander("⏱️ Holding Period"):
                            df_trades['hold'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.days
                            draw_stat("Avg Hold", f"{df_trades['hold'].mean():.2f} days")
                    with cr:
                        st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Strategy Equity Curve", color_discrete_sequence=['#3498db']), use_container_width=True)
                        st.plotly_chart(px.area(df_trades, x='exit_date', y=drawdown*100, title="Underwater Drawdown (%)", color_discrete_sequence=['#e74c3c']), use_container_width=True)

                with t3:
                    y_r = df_trades.groupby('year')['pnl_pct'].sum() * 100
                    st.plotly_chart(go.Figure(data=[go.Bar(x=y_r.index, y=y_r.values, text=y_r.values.round(1), texttemplate='%{text}%', textposition='outside', marker=dict(color='#3498db'))]).update_layout(title="Yearly Return (%)", template="plotly_dark"), use_container_width=True)

                with t4:
                    st.dataframe(df_trades[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl_pct', 'exit_reason']], use_container_width=True)
            else:
                st.warning("No trades found.")
    except Exception as e:
        st.error(f"Execution Error: {e}")