import streamlit as st
import pandas as pd
import requests
from io import StringIO

st.set_page_config(page_title="NSE/BSE Symbol Loader", layout="wide")

st.title("📊 NSE & BSE Full Symbol Auto Loader")

# ==========================================
# Load NSE Symbols
# ==========================================
@st.cache_data(ttl=86400)
def load_nse_symbols():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))

    # Clean column names
    df.columns = df.columns.str.strip()

    # Filter only EQ series
    df = df[df["SERIES"] == "EQ"]

    symbols = df["SYMBOL"].tolist()
    return sorted([s + ".NS" for s in symbols])


# ==========================================
# Load BSE Symbols (Yahoo Compatible)
# ==========================================
@st.cache_data(ttl=86400)
def load_bse_symbols():
    # Using same NSE list but converting to .BO for Yahoo
    nse_symbols = load_nse_symbols()
    bse_symbols = [s.replace(".NS", ".BO") for s in nse_symbols]
    return sorted(bse_symbols)


# ==========================================
# Combine NSE + BSE
# ==========================================
@st.cache_data(ttl=86400)
def load_all_symbols():
    nse = load_nse_symbols()
    bse = load_bse_symbols()
    return sorted(list(set(nse + bse)))


# ==========================================
# UI Section
# ==========================================
exchange = st.sidebar.radio("Select Exchange", ["NSE", "BSE", "All"])

if exchange == "NSE":
    symbols = load_nse_symbols()
elif exchange == "BSE":
    symbols = load_bse_symbols()
else:
    symbols = load_all_symbols()

symbol = st.sidebar.selectbox(
    "🔍 Search Stock Symbol",
    options=symbols
)

st.success(f"Selected Symbol: {symbol}")

st.write(f"Total Symbols Loaded: {len(symbols)}")