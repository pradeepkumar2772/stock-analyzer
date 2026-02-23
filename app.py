import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, date

# --------------------------------------------------
# 1. TRADE DATA STRUCTURE
# --------------------------------------------------
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


# --------------------------------------------------
# 2. RS-AV ENGINE
# --------------------------------------------------
def calculate_rsav(stock_df, benchmark_df, lookback=50):
    s_roc = stock_df['close'].pct_change(lookback) * 100
    s_vol = s_roc.rolling(window=lookback).std()
    s_net = s_roc - s_vol

    b_roc = benchmark_df['close'].pct_change(lookback) * 100
    b_vol = b_roc.rolling(window=lookback).std()
    b_net = b_roc - b_vol

    return s_net - b_net


# --------------------------------------------------
# 3. BACKTEST ENGINE (CORRECTED)
# --------------------------------------------------
def run_backtest(df, symbol, config, strategy_type, benchmark_df=None):

    trades = []
    active_trade = None
    slippage = (config.get('slippage_val', 0) / 100) if config.get('use_slippage', False) else 0

    df = df.copy()
    df['long_signal'] = False
    df['exit_signal'] = False

    # RS-AV
    if config.get('use_rsav', False) and benchmark_df is not None:
        df['rsav'] = calculate_rsav(df, benchmark_df, lookback=50)

    # Indicators
    df['ema_15_pk'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_20_pk'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_fast'] = df['close'].ewm(span=config.get('ema_fast', 20), adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config.get('ema_slow', 50), adjust=False).mean()
    df['ema_exit'] = df['close'].ewm(span=config.get('ema_exit', 30), adjust=False).mean()

    df['sma_200'] = df['close'].rolling(200).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['std_20'] = df['close'].rolling(20).std()
    df['upper_bb'] = df['sma_20'] + (df['std_20'] * 2)
    df['lower_bb'] = df['sma_20'] - (df['std_20'] * 2)
    df['bb_width'] = df['upper_bb'] - df['lower_bb']

    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    df['hhv'] = df['high'].rolling(config.get('hhv_period', 20)).max()
    df['llv'] = df['low'].rolling(config.get('hhv_period', 20)).min()
    df['neckline'] = df['high'].rolling(20).max()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = abs(delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    df['pole_return'] = df['close'].pct_change(10)
    df['is_pole'] = df['pole_return'] > 0.08
    df['flag_high'] = df['high'].rolling(3).max()
    df['flag_low'] = df['low'].rolling(3).min()

    # Remove warm-up rows
    df = df.dropna().copy()

    # ---------------- STRATEGIES ----------------
    if strategy_type == "PK Strategy (Positional)":
        df['long_signal'] = (df['close'].shift(1) < df['ema_20_pk'].shift(1)) & (df['close'] > df['ema_20_pk'])
        df['exit_signal'] = df['close'] < df['ema_15_pk']

    elif strategy_type == "RSI 60 Cross":
        df['long_signal'] = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        df['exit_signal'] = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)

    elif strategy_type == "EMA Ribbon":
        df['long_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['exit_signal'] = df['ema_fast'] < df['ema_exit']

    else:
        df['long_signal'] = False
        df['exit_signal'] = False

    # ---------------- EXECUTION LOOP ----------------
    for i in range(1, len(df)):

        current = df.iloc[i]
        prev = df.iloc[i - 1]

        market_ok = True
        if config.get('use_rsav', False) and 'rsav' in df.columns:
            market_ok = current['rsav'] >= config.get('rsav_trigger', -0.5)

        if active_trade:

            sl_price = active_trade.entry_price * (1 - config.get('sl_val', 0) / 100)
            tp_price = active_trade.entry_price * (1 + config.get('tp_val', 0) / 100)

            sl_hit = config.get('use_sl', False) and current['low'] <= sl_price
            tp_hit = config.get('use_tp', False) and current['high'] >= tp_price

            if sl_hit or tp_hit or prev['exit_signal']:

                if sl_hit:
                    exit_price = sl_price
                    reason = "Stop Loss"
                elif tp_hit:
                    exit_price = tp_price
                    reason = "Target Profit"
                else:
                    exit_price = current['open']
                    reason = "Signal Exit"

                exit_price *= (1 - slippage)

                active_trade.exit_price = exit_price
                active_trade.exit_date = current.name
                active_trade.exit_reason = reason
                active_trade.pnl_pct = (
                    (exit_price - active_trade.entry_price)
                    / active_trade.entry_price
                )

                trades.append(active_trade)
                active_trade = None

        elif prev['long_signal'] and market_ok:

            entry_price = current['open'] * (1 + slippage)

            active_trade = Trade(
                symbol=symbol,
                direction="Long",
                entry_date=current.name,
                entry_price=entry_price
            )

    return trades, df


# --------------------------------------------------
# 4. UI
# --------------------------------------------------
st.set_page_config(layout="wide", page_title="Strategy Lab Pro")

st.sidebar.title("🎗️ Strategy Engine")
symbol = st.sidebar.text_input("Symbol", value="BRITANNIA.NS").upper()

strategies_list = [
    "PK Strategy (Positional)",
    "RSI 60 Cross",
    "EMA Ribbon"
]

strat_choice = st.sidebar.selectbox("Select Strategy", strategies_list)

tf_map = {"Daily": "1d", "Weekly": "1wk"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=0)

capital = st.sidebar.number_input("Initial Capital", value=1000.0)
start_str = st.sidebar.text_input("Start Date", value="2005-01-01")
end_str = st.sidebar.text_input("End Date", value=date.today().strftime('%Y-%m-%d'))

st.sidebar.divider()

use_sl = st.sidebar.toggle("Stop Loss", True)
sl_val = st.sidebar.slider("SL %", 0.5, 15.0, 5.0)
use_tp = st.sidebar.toggle("Target Profit", True)
tp_val = st.sidebar.slider("TP %", 1.0, 100.0, 25.0)
use_slip = st.sidebar.toggle("Slippage", True)
slip_val = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1)

config = {
    "use_sl": use_sl,
    "sl_val": sl_val,
    "use_tp": use_tp,
    "tp_val": tp_val,
    "use_slippage": use_slip,
    "slippage_val": slip_val
}

run_btn = st.sidebar.button("🚀 Run Backtest")

# --------------------------------------------------
# 5. EXECUTION
# --------------------------------------------------
if run_btn:

    data = yf.download(symbol, start=start_str, end=end_str, interval=tf_map[selected_tf], auto_adjust=True)

    if not data.empty:

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [col.lower() for col in data.columns]

        trades, _ = run_backtest(data, symbol, config, strat_choice)

        if trades:

            df_trades = pd.DataFrame([vars(t) for t in trades])
            df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
            df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
            df_trades['equity'] = capital * (1 + df_trades['pnl_pct']).cumprod()

            wins = df_trades[df_trades['pnl_pct'] > 0]
            losses = df_trades[df_trades['pnl_pct'] <= 0]

            total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100

            duration = df_trades['exit_date'].max() - df_trades['entry_date'].min()
            years_v = max(duration.days / 365.25, 0.1)
            cagr = (((df_trades['equity'].iloc[-1] / capital) ** (1 / years_v)) - 1) * 100

            peak = df_trades['equity'].cummax()
            drawdown = (df_trades['equity'] - peak) / peak
            mdd = drawdown.min() * 100

            # Proper expectancy
            win_rate = len(wins) / len(df_trades)
            avg_win = wins['pnl_pct'].mean() if not wins.empty else 0
            avg_loss = losses['pnl_pct'].mean() if not losses.empty else 0
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

            # Safe Sharpe
            if len(df_trades) > 1 and df_trades['pnl_pct'].std() != 0:
                sharpe = (df_trades['pnl_pct'].mean() / df_trades['pnl_pct'].std()) * np.sqrt(252)
            else:
                sharpe = 0.0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return (%)", f"{total_ret:.2f}%")
            col2.metric("CAGR (%)", f"{cagr:.2f}%")
            col3.metric("Max Drawdown (%)", f"{mdd:.2f}%")
            col4.metric("Win Rate (%)", f"{win_rate*100:.2f}%")

            st.metric("Expectancy (per trade)", f"{expectancy*100:.2f}%")
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

            st.plotly_chart(
                px.line(df_trades, x='exit_date', y='equity', title="Equity Curve"),
                use_container_width=True
            )

        else:
            st.warning("No trades found.")
    else:
        st.error("No data found.")