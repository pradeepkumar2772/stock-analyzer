import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date, timedelta

# --- 1. CORE DATA STRUCTURE (Identical) ---
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

# --- RS-AV CALCULATION ENGINE (Identical) ---
def calculate_rsav(stock_df, benchmark_df, lookback=50):
    s_roc = stock_df['close'].pct_change(lookback) * 100
    s_vol = s_roc.rolling(window=lookback).std()
    s_net = s_roc - s_vol
    b_roc = benchmark_df['close'].pct_change(lookback) * 100
    b_vol = b_roc.rolling(window=lookback).std()
    b_net = b_roc - b_vol
    return s_net - b_net

# --- 2. MULTI-STRATEGY ENGINE (Identical) ---
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):
    trades = []
    active_trade = None
    slippage = (config['slippage_val'] / 100) if config['use_slippage'] else 0
    
    # Initialize columns to prevent KeyError
    df['long_signal'] = False
    df['exit_signal'] = False
    
    if config.get('use_rsav', False) and benchmark_df is not None:
        df['rsav'] = calculate_rsav(df, benchmark_df, lookback=50)

    # Indicator Calculations
    df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['std_20'] = df['close'].rolling(window=20).std()
    df['upper_bb'] = df['sma_20'] + (df['std_20'] * 2)
    df['lower_bb'] = df['sma_20'] - (df['std_20'] * 2)
    df['bb_width'] = df['upper_bb'] - df['lower_bb']
    
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    h_per = config.get('hhv_period', 20)
    df['hhv'] = df['high'].rolling(window=h_per).max()
    df['llv'] = df['low'].rolling(window=h_per).min()
    df['neckline'] = df['high'].rolling(window=20).max()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    df['pole_return'] = df['close'].pct_change(periods=10)
    df['is_pole'] = df['pole_return'] > 0.08
    df['flag_high'] = df['high'].rolling(window=3).max()
    df['flag_low'] = df['low'].rolling(window=3).min()

    # Strategy Switch Logic
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = (df['close'] < df['ema_15_pk'])
    elif strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)
    elif strategy_type == "EMA Ribbon":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = (df['ema_fast'] < df['ema_exit'])
    elif strategy_type == "Flags & Pennants":
        df['long_signal'] = df['is_pole'].shift(3) & (df['close'] > df['flag_high'].shift(1))
        df['exit_signal'] = (df['close'] < df['flag_low'].shift(1))
    elif strategy_type == "Bollinger Squeeze Breakout":
        is_sqz = df['bb_width'] <= df['bb_width'].rolling(window=20).min()
        df['long_signal'] = is_sqz.shift(1) & (df['close'] > df['upper_bb'])
        df['exit_signal'] = (df['close'] < df['sma_20'])
    elif strategy_type == "EMA & RSI Synergy":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['rsi'] > 60)
        df['exit_signal'] = (df['close'] < df['ema_exit']) | (df['rsi'] < 40)
    elif strategy_type == "RSI Divergence":
        price_ll = df['low'] < df['low'].shift(10); rsi_hl = df['rsi'] > df['rsi'].shift(10)
        df['long_signal'] = price_ll & rsi_hl & (df['close'] > df['high'].shift(1))
        df['exit_signal'] = (df['high'] > df['high'].shift(10)) & (df['rsi'] < df['rsi'].shift(10))
    elif strategy_type == "BB & RSI Exhaustion":
        df['long_signal'] = (df['low'] <= df['lower_bb']) & (df['rsi'] < 30)
        df['exit_signal'] = (df['close'] >= df['sma_20']) | (df['rsi'] > 50)
    elif strategy_type == "Breakaway Gap Momentum":
        df['long_signal'] = (df['open'] > df['high'].shift(1)) & (df['close'].shift(1) >= df['hhv'].shift(2) * 0.98)
        df['exit_signal'] = (df['close'] < df['low'].shift(1))
    elif strategy_type == "ATR Band Breakout":
        u_atr = df['sma_20'] + df['atr']; l_atr = df['sma_20'] - df['atr']
        df['long_signal'] = (df['close'] > u_atr) & (df['close'].shift(1) <= u_atr.shift(1))
        df['exit_signal'] = (df['close'] < l_atr) & (df['close'].shift(1) >= l_atr.shift(1))
    elif strategy_type == "HHV/LLV Breakout":
        df['long_signal'] = (df['close'] > df['hhv'].shift(1)); df['exit_signal'] = (df['close'] < df['llv'].shift(1))
    elif strategy_type == "Double Bottom Breakout":
        df['long_signal'] = (df['close'] > df['neckline'].shift(1)); df['exit_signal'] = (df['close'] < df['ema_exit'])
    elif strategy_type == "Fibonacci 61.8% Retracement":
        uptrend = df['close'] > df['sma_200']; fib = df['hhv'] - ((df['hhv'] - df['llv']) * 0.618)
        df['long_signal'] = uptrend & (df['low'] <= fib) & (df['close'] > df['high'].shift(1))
        df['exit_signal'] = df['close'] < df['llv'].shift(1)
    elif strategy_type == "Relative Strength Play":
        stock_ret = df['close'].pct_change(periods=55)
        df['long_signal'] = (stock_ret > 0) & (df['close'] > df['ema_fast']); df['exit_signal'] = (df['close'] < df['ema_slow'])

    for i in range(1, len(df)):
        current = df.iloc[i]; prev = df.iloc[i-1]
        market_ok = True
        if config.get('use_rsav', False) and 'rsav' in df.columns:
            market_ok = current['rsav'] >= config.get('rsav_trigger', -0.5)
            
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
        elif prev['long_signal'] and market_ok:
            active_trade = Trade(symbol=symbol, direction="Long", entry_date=current.name, entry_price=current['open'] * (1 + slippage))
    return trades, df

# --- NEW: OPTIMIZER ENGINE (Added Separately) ---
def run_pk_optimizer(data, symbol, config):
    results = []
    ema_range = range(3, 61, 2)
    total_steps = len(ema_range) * len(ema_range)
    current_step = 0
    progress_bar = st.progress(0)
    
    for entry_ema in ema_range:
        for exit_ema in ema_range:
            current_step += 1
            if entry_ema <= exit_ema:
                continue
            
            df_opt = data.copy()
            df_opt['ema_20_pk'] = df_opt['close'].ewm(span=entry_ema, adjust=False).mean()
            df_opt['ema_15_pk'] = df_opt['close'].ewm(span=exit_ema, adjust=False).mean()
            
            trades, _ = run_backtest(df_opt, symbol, config, "PK Strategy (Positional)")
            
            if trades:
                df_res = pd.DataFrame([vars(t) for t in trades])
                df_res['equity'] = 1000 * (1 + df_res['pnl_pct']).cumprod()
                total_ret = (df_res['equity'].iloc[-1] / 1000 - 1) * 100
                mdd = ((df_res['equity'] - df_res['equity'].cummax()) / df_res['equity'].cummax()).min() * 100
                rf = total_ret / abs(mdd*100) if mdd != 0 else total_ret
                results.append({"Entry EMA": entry_ema, "Exit EMA": exit_ema, "Return %": round(total_ret, 2), "Max DD %": round(mdd*100, 2), "Recovery Factor": round(rf, 2), "Trades": len(trades)})
            
            if current_step % 50 == 0:
                progress_bar.progress(current_step / total_steps)
    
    progress_bar.empty()
    return pd.DataFrame(results)

# --- 3. UI STYLING (Identical) ---
st.set_page_config(layout="wide", page_title="Strategy Lab Pro")
st.markdown("<style>.stMetric { background-color: #1a1c24; padding: 18px; border-radius: 8px; border: 1px solid #2d2f3b; } .report-table { width: 100%; border-collapse: collapse; margin-top: 10px; } .report-table td { border: 1px solid #2d2f3b; padding: 10px; text-align: center; color: #fff; font-size: 0.85rem; } .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d2f3b; } .stat-label { color: #999; font-size: 0.85rem; } .stat-value { color: #fff; font-weight: 600; font-size: 0.85rem; }</style>", unsafe_allow_html=True)

def draw_stat(label, value):
    st.markdown(f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()
strategies_list = ["PK Strategy (Positional)", "RSI 60 Cross", "EMA Ribbon", "Flags & Pennants", "Bollinger Squeeze Breakout", "Breakaway Gap Momentum", "EMA & RSI Synergy", "RSI Divergence", "BB & RSI Exhaustion", "Relative Strength Play", "ATR Band Breakout", "HHV/LLV Breakout", "Double Bottom Breakout", "Fibonacci 61.8% Retracement"]
strat_choice = st.sidebar.selectbox("Select Strategy", strategies_list)
tf_map = {"1 Minute": "1m", "2 Minutes": "2m", "3 Minutes": "3m", "5 Minutes": "5m", "10 Minutes": "10m", "15 Minutes": "15m", "30 Minutes": "30m", "1 Hour": "1h", "Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=8)
capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="1999-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

st.sidebar.divider()
st.sidebar.subheader("🛡️ Market Filter (RS-AV)")
use_rsav = st.sidebar.toggle("Enable RS-AV Filter", False)
rsav_trig = st.sidebar.number_input("RS-AV Trigger", value=-0.5, step=0.1)

config = {'ema_fast': 20, 'ema_slow': 50, 'ema_exit': 30, 'hhv_period': 20, 'use_rsav': use_rsav, 'rsav_trigger': rsav_trig}
st.sidebar.divider()
use_sl = st.sidebar.toggle("Stop Loss", True); config['sl_val'] = st.sidebar.slider("SL %", 0.5, 15.0, 5.0) if use_sl else 0; config['use_sl'] = use_sl
use_tp = st.sidebar.toggle("Target Profit", True); config['tp_val'] = st.sidebar.slider("TP %", 1.0, 100.0, 25.0) if use_tp else 0; config['use_tp'] = use_tp
use_slip = st.sidebar.toggle("Slippage", True); config['slippage_val'] = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1) if use_slip else 0; config['use_slippage'] = use_slip

# --- 5. EXECUTION BUTTONS ---
col_run1, col_run2, col_run3 = st.sidebar.columns(3)
run_single = col_run1.button("🚀 Run Backtest")
run_arena = col_run2.button("🏟️ Run Arena")
run_opt = col_run3.button("🎯 Run Optimizer")

if run_single:
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
        bench_data = yf.download("^NSEI", start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True) if use_rsav else None
        
        if not data.empty:
            for d in [data, bench_data]:
                if d is not None:
                    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                    d.columns = [str(col).lower() for col in d.columns]
            if use_rsav and bench_data is not None:
                common_idx = data.index.intersection(bench_data.index)
                data, bench_data = data.loc[common_idx], bench_data.loc[common_idx]

            trades, processed_df = run_backtest(data.copy(), symbol, config, strat_choice, bench_data)
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
                df_trades['hold'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.days

                t1, t2, t3, t4 = st.tabs(["Quick Stats", "Statistics", "Charts", "Trade Details"])
                with t1:
                    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                    r1c1.metric("Total Returns (%)", f"{total_ret:.2f}%"); r1c2.metric("Max Drawdown(MDD)", f"{mdd:.2f}%"); r1c3.metric("Win Ratio", f"{(len(wins)/len(df_trades)*100):.2f}%"); r1c4.metric("Total Trades", len(df_trades))
                    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                    r2c1.metric("Initial Capital", f"{capital:,.2f}"); r2c2.metric("Final Capital", f"{df_trades['equity'].iloc[-1]:,.2f}"); r2c3.metric("CAGR", f"{cagr:.2f}%"); r2c4.metric("Avg Return/Trade", f"{(df_trades['pnl_pct'].mean()*100):.2f}%")
                    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
                    r3c1.metric("Risk-Reward Ratio", f"{rr:.2f}"); r3c2.metric("Expectancy", f"{exp:.2f}"); r3c3.metric("Sharpe Ratio", f"{sharpe:.2f}"); r3c4.metric("Calmar Ratio", f"{calmar:.2f}")
                with t2:
                    cl, cr = st.columns([1, 2.5])
                    with cl:
                        with st.expander("📊 Backtest Details", expanded=True):
                            draw_stat("Scrip", symbol); draw_stat("Start Date", df_trades['entry_date'].min().strftime('%d-%b %y')); draw_stat("Duration", f"{duration.days // 365} Years, {(duration.days % 365) // 30} Months")
                        with st.expander("📈 Return"):
                            draw_stat("Total Return", f"{total_ret:.2f} %"); draw_stat("CAGR", f"{cagr:.2f}%"); draw_stat("Average Return Per Trade", f"{df_trades['pnl_pct'].mean()*100:.2f} %")
                        with st.expander("📉 Drawdown"):
                            draw_stat("Maximum Drawdown", f"{mdd:.2f} %"); draw_stat("Average Drawdown", f"{drawdown.mean()*100:.2f} %")
                        with st.expander("🏆 Performance"):
                            draw_stat("Win Rate", f"{(len(wins)/len(df_trades)*100):.2f} %"); draw_stat("Risk Reward Ratio", f"{rr:.2f}"); draw_stat("Expectancy", f"{exp:.2f}")
                        with st.expander("🔍 Trade Characteristics"):
                            draw_stat("Total Number of Trades", len(df_trades)); draw_stat("Winning Streak", f"{max_w_s}.00"); draw_stat("Lossing Streak", f"{max_l_s}.00")
                        with st.expander("🛡️ Risk-Adjusted Metrics"):
                            draw_stat("Sharpe Ratio", f"{sharpe:.2f}"); draw_stat("Calmar Ratio", f"{calmar:.2f}")
                        with st.expander("⏱️ Holding Period"):
                            draw_stat("Max Holding Period", f"{df_trades['hold'].max()} days"); draw_stat("Average Holding Period", f"{df_trades['hold'].mean():.2f} days")
                        with st.expander("🔥 Streak"):
                            draw_stat("Win Streak", max_w_s); draw_stat("Loss Streak", max_l_s)
                    with cr:
                        st.plotly_chart(px.line(df_trades, x='exit_date', y='equity', title="Strategy Equity Curve"), use_container_width=True)
            else: st.warning("No trades found.")
    except Exception as e: st.error(f"Error: {e}")

elif run_arena:
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            
            bench_data = None
            if use_rsav:
                bench_data = yf.download("^NSEI", start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
                if not bench_data.empty:
                    if isinstance(bench_data.columns, pd.MultiIndex): bench_data.columns = bench_data.columns.get_level_values(0)
                    bench_data.columns = [str(col).lower() for col in bench_data.columns]
                    common_idx = data.index.intersection(bench_data.index)
                    data = data.loc[common_idx]
                    bench_data = bench_data.loc[common_idx]

            arena_results = []
            combined_fig = go.Figure()
            progress_bar = st.progress(0)
            
            for idx, s_name in enumerate(strategies_list):
                trades, _ = run_backtest(data.copy(), symbol, config, s_name, benchmark_df=bench_data)
                
                total_ret = 0.0; mdd_val = 0.0; pf = 0.0; rf = 0.0; win_r = 0.0; trade_count = len(trades)
                
                if trades:
                    df_res = pd.DataFrame([vars(t) for t in trades])
                    df_res['equity'] = capital * (1 + df_res['pnl_pct']).cumprod()
                    total_ret = (df_res['equity'].iloc[-1] / capital - 1) * 100
                    mdd_val = ((df_res['equity'] - df_res['equity'].cummax()) / df_res['equity'].cummax()).min() * 100
                    win_r = (len(df_res[df_res['pnl_pct'] > 0]) / len(df_res)) * 100
                    p_sum = df_res[df_res['pnl_pct'] > 0]['pnl_pct'].sum()
                    l_sum = abs(df_res[df_res['pnl_pct'] <= 0]['pnl_pct'].sum())
                    pf = p_sum / l_sum if l_sum != 0 else p_sum
                    rf = total_ret / abs(mdd_val) if mdd_val != 0 else total_ret
                    combined_fig.add_trace(go.Scatter(x=df_res['exit_date'], y=df_res['equity'], name=s_name))
                
                arena_results.append({
                    "Strategy": s_name, 
                    "Total Return %": round(total_ret, 2), 
                    "Max DD %": round(mdd_val, 2), 
                    "Profit Factor": round(pf, 2),
                    "Recovery Factor": round(rf, 2),
                    "Win %": round(win_r, 2), 
                    "Trades": trade_count
                })
                progress_bar.progress((idx + 1) / len(strategies_list))
            
            st.subheader("🏟️ Strategy Arena Leaderboard")
            res_df = pd.DataFrame(arena_results).sort_values(by="Total Return %", ascending=False)
            
            def color_leaderboard(val, column):
                if column == "Max DD %":
                    return "color: #ff4b4b;" if val < -15 else ""
                if column in ["Total Return %", "Recovery Factor", "Profit Factor"]:
                    if column == "Profit Factor": return "color: #00ff00; font-weight: bold;" if val > 2.0 else ""
                    return "color: #00ff00;" if val > 50 else ""
                return ""

            st.dataframe(
                res_df.style.apply(lambda x: [color_leaderboard(v, x.name) for v in x]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Total Return %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Max DD %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Recovery Factor": st.column_config.ProgressColumn(min_value=0, max_value=float(res_df['Recovery Factor'].max() if not res_df.empty else 1))
                }
            )
            
            st.plotly_chart(combined_fig.update_layout(title="Arena Equity Comparison", template="plotly_dark"), use_container_width=True)
    except Exception as e: st.error(f"Arena Error: {e}")

elif run_opt:
    try:
        data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            
            # Baseline (current 20/15)
            b_trades, _ = run_backtest(data.copy(), symbol, config, "PK Strategy (Positional)")
            b_ret = 0.0
            if b_trades:
                b_df = pd.DataFrame([vars(t) for t in b_trades])
                b_df['equity'] = capital * (1 + b_df['pnl_pct']).cumprod()
                b_ret = (b_df['equity'].iloc[-1] / capital - 1) * 100

            opt_results = run_pk_optimizer(data, symbol, config)
            
            if not opt_results.empty:
                st.subheader(f"🎯 Optimization Summary: {symbol}")
                best_row = opt_results.loc[opt_results['Recovery Factor'].idxmax()]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Return (20/15)", f"{b_ret:.2f}%")
                c2.metric("Best Found Return", f"{best_row['Return %']:.2f}%", delta=f"{best_row['Return %'] - b_ret:.2f}%")
                c3.metric("Best EMA Combo", f"{best_row['Entry EMA']} / {best_row['Exit EMA']}")

                st.divider()
                st.write("### 🗺️ Strategy Parameter Heatmap")
                
                
                st.dataframe(
                    opt_results.sort_values(by="Recovery Factor", ascending=False),
                    column_config={
                        "Recovery Factor": st.column_config.ProgressColumn(
                            min_value=0, 
                            max_value=float(opt_results['Recovery Factor'].max())
                        ),
                        "Return %": st.column_config.NumberColumn(format="%.2f%%"),
                        "Max DD %": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No trades found during optimization.")
    except Exception as e:
        st.error(f"Optimizer Error: {e}")