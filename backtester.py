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
# ADD EMA
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
# BACKTEST
# ---------------------------
def backtest(data, initial_capital=100000,
             brokerage_pct=0.0005,
             slippage_pct=0.0005,
             stop_loss_pct=0.02):

    capital = initial_capital
    position = 0
    entry_price = 0
    equity_curve = []
    trade_results = []
    trade_log = []

    for i in range(len(data)):
        price = data['Close'].iloc[i]
        date = data.index[i]

        # ENTRY
        if data['Position'].iloc[i] == 1 and position == 0:

            entry_price = price * (1 + slippage_pct)
            qty = capital / entry_price
            capital -= capital * brokerage_pct

            position = qty
            capital = 0

            trade_log.append({"Entry Date": date,
                              "Entry Price": entry_price})

        # EXIT
        elif data['Position'].iloc[i] == -1 and position > 0:

            exit_price = price * (1 - slippage_pct)
            capital = position * exit_price
            capital -= capital * brokerage_pct

            pnl_pct = (exit_price - entry_price) / entry_price * 100
            trade_results.append(pnl_pct)

            trade_log[-1].update({
                "Exit Date": date,
                "Exit Price": exit_price,
                "PnL %": round(pnl_pct, 2)
            })

            position = 0

        # STOP LOSS
        if position > 0:
            if price <= entry_price * (1 - stop_loss_pct):

                exit_price = price * (1 - slippage_pct)
                capital = position * exit_price
                capital -= capital * brokerage_pct

                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trade_results.append(pnl_pct)

                trade_log[-1].update({
                    "Exit Date": date,
                    "Exit Price": exit_price,
                    "PnL %": round(pnl_pct, 2)
                })

                position = 0

        equity = capital + position * price
        equity_curve.append(equity)

    data['Equity'] = equity_curve
    final_value = equity_curve[-1]

    return data, final_value, trade_results, pd.DataFrame(trade_log)


# ---------------------------
# PERFORMANCE METRICS
# ---------------------------
def performance_metrics(data, trade_results, initial_capital):

    final_value = data['Equity'].iloc[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    # CAGR
    days = (data.index[-1] - data.index[0]).days
    years = days / 365
    cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Win Rate
    wins = len([x for x in trade_results if x > 0])
    total_trades = len(trade_results)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # Drawdown
    cumulative_max = data['Equity'].cummax()
    drawdown = (data['Equity'] - cumulative_max) / cumulative_max
    max_dd = drawdown.min() * 100

    return {
        "Final Value": round(final_value, 2),
        "Total Return %": round(total_return, 2),
        "CAGR %": round(cagr, 2),
        "Win Rate %": round(win_rate, 2),
        "Max Drawdown %": round(max_dd, 2),
        "Total Trades": total_trades
    }