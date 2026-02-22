import yfinance as yf

def fetch_data(symbol, start, end, interval):
    data = yf.download(symbol, start=start, end=end, interval=interval)
    data.dropna(inplace=True)
    return data