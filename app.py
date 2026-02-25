import streamlit as st
import yfinance as yf
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Stock News Streamliner", page_icon="📈", layout="wide")

st.title("📈 Real-Time Stock News & Insights")
st.markdown("---")

# Sidebar for User Input
st.sidebar.header("Settings")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., TSLA, NVDA):", value="AAPL").upper()

if ticker_symbol:
    try:
        # Initialize Ticker
        stock = yf.Ticker(ticker_symbol)
        
        # 1. Show Quick Stock Stats
        info = stock.fast_info
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Current Price", f"${info['last_price']:.2f}")
        col_b.metric("Day Change", f"{info['day_change']:.2f}%")
        col_c.metric("Currency", info['currency'])

        st.write("## Recent Headlines")

        # 2. Fetch News with Error Handling
        news_list = stock.news

        if not news_list:
            st.warning(f"No recent news found for {ticker_symbol}.")
        else:
            for article in news_list:
                # Use .get() with defaults to prevent "KeyError"
                title = article.get('title', 'Headline not available')
                link = article.get('link', '#')
                publisher = article.get('publisher', 'Unknown Source')
                
                # Convert timestamp to readable date
                timestamp = article.get('providerPublishTime', 0)
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M') if timestamp else "N/A"

                with st.container():
                    col1, col2 = st.columns([1, 4])
                    
                    # Safe Thumbnail Fetching
                    with col1:
                        thumbnail_data = article.get("thumbnail", {})
                        resolutions = thumbnail_data.get("resolutions", [])
                        if resolutions:
                            img_url = resolutions[0].get("url")
                            st.image(img_url, use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/150?text=No+Image", use_container_width=True)

                    # Content Display
                    with col2:
                        st.subheader(title)
                        st.caption(f"📅 {date_str} | 🏢 Source: {publisher}")
                        st.markdown(f"[Read Full Article]({link})")
                    
                    st.divider()

    except Exception as e:
        st.error(f"Could not fetch data for '{ticker_symbol}'. Please check the ticker symbol and try again.")
        st.info(f"Technical details: {e}")

else:
    st.info("Please enter a stock ticker in the sidebar to begin.")