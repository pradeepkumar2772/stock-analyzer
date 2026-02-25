import streamlit as st
import yfinance as yf

# App Title
st.set_page_config(page_title="Stock News Streamliner", page_icon="📈")
st.title("🗞️ Real-Time Stock News")

# Sidebar for User Input
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()

if ticker_symbol:
    try:
        # Fetch data using yfinance
        stock = yf.Ticker(ticker_symbol)
        news_list = stock.news
        
        st.subheader(f"Latest news for {ticker_symbol}")

        if not news_list:
            st.warning("No recent news found for this ticker.")
        else:
            for article in news_list:
                with st.container():
                    # Display Article Layout
                    col1, col2 = st.columns([1, 3])
                    
                    # Thumbnail (if available)
                    with col1:
                        if "thumbnail" in article and "resolutions" in article["thumbnail"]:
                            st.image(article["thumbnail"]["resolutions"][0]["url"])
                        else:
                            st.image("https://via.placeholder.com/150", caption="No Image")

                    # Title and Link
                    with col2:
                        st.markdown(f"### [{article['title']}]({article['link']})")
                        st.write(f"**Source:** {article['publisher']} | **Published:** {article['providerPublishTime']}")
                    
                    st.divider()

    except Exception as e:
        st.error(f"Error fetching data: {e}")