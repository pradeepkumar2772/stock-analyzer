import streamlit as st
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Global Stock News", layout="wide")
st.title("🌍 Global Stock News Streamliner")

ticker_symbol = st.sidebar.text_input("Enter Ticker (e.g., BRITANNIA.NS, TSLA):", value="BRITANNIA.NS").upper()

if ticker_symbol:
    try:
        stock = yf.Ticker(ticker_symbol)
        
        # --- ROBUST PRICE FETCHING ---
        current_price = None
        try:
            # Attempt 1: Fast Info (Fastest)
            current_price = stock.fast_info.get('last_price')
        except:
            pass
            
        if current_price is None:
            # Attempt 2: History fallback (Reliable for NSE/BSE)
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]

        # Display Price Header
        if current_price:
            st.metric(label=f"{ticker_symbol} Price", value=f"₹{current_price:.2f}" if ".NS" in ticker_symbol else f"${current_price:.2f}")
        else:
            st.warning("⚠️ Price data currently unavailable, but fetching news...")

        # --- SAFE NEWS FETCHING ---
        news_list = stock.news
        if not news_list:
            st.info(f"No recent news articles found for {ticker_symbol}.")
        else:
            for article in news_list:
                # Use .get() to avoid the "KeyError" on 'title' or other fields
                title = article.get('title', 'No Headline')
                link = article.get('link', '#')
                source = article.get('publisher', 'Financial News')
                
                with st.expander(f"📰 {title}"):
                    st.write(f"**Source:** {source}")
                    st.write(f"[Read Full Story]({link})")
                    
    except Exception as e:
        st.error(f"Could not load data for {ticker_symbol}. Check if the symbol is correct.")
        st.caption(f"Error details: {e}")