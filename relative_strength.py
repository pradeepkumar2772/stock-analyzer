import yfinance as yf
import pandas as pd


def relative_strength(stock_symbol, benchmark_symbol, start, end):

    stock = yf.download(stock_symbol, start=start, end=end)['Close']
    benchmark = yf.download(benchmark_symbol, start=start, end=end)['Close']

    rs = stock / benchmark
    rs = rs / rs.iloc[0]  # Normalize

    rs_df = pd.DataFrame(rs)
    rs_df.columns = ['RS']

    return rs_df