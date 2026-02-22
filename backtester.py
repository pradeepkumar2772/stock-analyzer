import pandas as pd
import numpy as np
import yfinance as yf


# ---------------------------
# DATA FETCH
# ---------------------------
def fetch_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data.dropna(inplace=True)
    return data


# ---------------------------
# INDICATORS
# ---------------------------
def add_ema(data, short=20, long=50):
    data['EMA_SHORT'] = data['Close'].ewm(span=short, adjust=False).mean()
    data['EMA_LONG'] = data['Close'].ewm(span=long, adjust=False).mean()
    return data


# ---------------------------
# SIGNAL GENERATION
# ---------------------------
def generate_signals(data):
    data['Signal'] = 0
    data.loc[data['EMA_SHORT'] > data['EMA_LONG'], 'Signal'] = 1
    data['Position'] = data['Signal'].diff()
    return data


# ---------------------------
# BACKTEST ENGINE
# ---------------------------
def backtest(
    data,
    initial_capital=100000,
    brokerage_pct=0.0005,
    slippage_pct=0.0005,
    stop_loss_pct=0.02
):

    capital = initial_capital
    position = 0
    entry_price = 0
    equity_curve = []
    trades = []
    trade_log = []

    for i in range(len(data)):

        price = data['Close'].iloc[i]
        date = data.index[i]

        # ENTRY
        if data['Position'].iloc[i] == 1 and position == 0:

            entry_price = price * (1 + slippage_pct)
            qty = capital / entry_price
            cost = capital * brokerage_pct

            capital -= cost
            position = qty
            capital = 0

            trade_log.append({
                "Entry Date": date,
                "Entry Price": entry_price
            })

        # EXIT (Crossover)
        elif data['Position'].iloc[i] == -1 and position > 0:

            exit_price = price * (1 - slippage_pct)
            capital = position * exit_price
            cost = capital * brokerage_pct
            capital -= cost

            pnl = capital - initial_capital

            trade_log[-1].update({
                "Exit Date": date,
                "Exit Price": exit_price,
                "Capital After Trade": capital
            })

            trades.append(capital)
            position = 0

        # STOP LOSS
        if position > 0:
            if price <= entry_price * (1 - stop_loss_pct):

                exit_price = price * (1 - slippage_pct)
                capital = position * exit_price
                cost = capital * brokerage_pct
                capital -= cost

                trade_log[-1].update({
                    "Exit Date": date,
                    "Exit Price": exit_price,
                    "Capital After Trade": capital
                })

                trades.append(capital)
                position = 0

        equity = capital + position * price
        equity_curve.append(equity)

    data['Equity'] = equity_curve
    final_value = equity_curve[-1]

    return data, final_value, trades, pd.DataFrame(trade_log)


# ---------------------------
# PERFORMANCE METRICS
# ---------------------------
def performance_metrics(data, initial_capital):

    final_value = data['Equity'].iloc[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    daily_returns = data['Equity'].pct_change().dropna()

    sharpe = 0
    if daily_returns.std() != 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

    cumulative_max = data['Equity'].cummax()
    drawdown = (data['Equity'] - cumulative_max) / cumulative_max
    max_dd = drawdown.min() * 100

    return {
        "Final Value": round(final_value, 2),
        "Total Return %": round(total_return, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown %": round(max_dd, 2)
    }


# ---------------------------
# STRATEGY SUGGESTION
# ---------------------------
def strategy_suggestion(metrics):

    score = (
        metrics["Total Return %"] * 0.5
        - abs(metrics["Max Drawdown %"]) * 0.3
        + metrics["Sharpe Ratio"] * 20
    )

    if score > 50:
        return "Strong Trend Strategy"
    elif score > 20:
        return "Moderate Strategy"
    else:
        return "Avoid / Needs Optimization"