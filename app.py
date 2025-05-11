import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
from utils.stock_analysis import calculate_technical_indicators
from utils.news_sentiment import get_news_sentiment
from utils.prediction import predict_stock_price
import requests
import time
from functools import lru_cache  # For caching

# ✅ Monkey patch yfinance to use browser headers (bypass rate-limiting)
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive"
})
yf.shared._requests = session  # Monkey-patch session used by yfinance


def fetch_stock_data_with_retries(ticker, period, retries=3, backoff_factor=2):
    """
    Fetch stock data with retries and exponential backoff.
    """
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            # Check if data is empty
            if hist.empty:
                raise ValueError("No data found or request was rate-limited.")

            return hist  # Return data if successful

        except Exception as e:
            if attempt < retries - 1:
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)  # Exponential backoff
            else:
                raise e  # Re-raise error if all retries fail


@lru_cache(maxsize=128)
def cached_fetch_stock_data(ticker, period):
    """
    Fetch and cache stock data using LRU cache.
    """
    return fetch_stock_data_with_retries(ticker, period)


# Page config
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    st.title("📈 Stock Prediction & Analysis")

    # Sidebar
    st.sidebar.header("Settings")
    ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS for NSE stocks)", "RELIANCE.NS")
    period = st.sidebar.selectbox(
        "Select Time Period",
        ("1mo", "3mo", "6mo", "1y", "2y", "5y")
    )

    if st.sidebar.button("Analyze"):
        try:
            with st.spinner('Fetching stock data...'):
                # Enhanced: Fetch stock data with retries and caching
                hist = cached_fetch_stock_data(ticker, period)
                if hist.empty:
                    st.error("No data found or request was rate-limited. Please try again later.")
                    return

            # Create three columns for stock metrics
            col1, col2, col3 = st.columns(3)

            # Current Stock Info
            with col1:
                current_price = hist['Close'].iloc[-1]
                price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                price_change_pct = (price_change / hist['Close'].iloc[-2]) * 100

                st.metric(
                    "Current Price",
                    f"₹{current_price:.2f}",
                    f"{price_change_pct:.2f}%"
                )

            with col2:
                st.metric(
                    "Volume",
                    f"{hist['Volume'].iloc[-1]:,.0f}",
                    f"{((hist['Volume'].iloc[-1] - hist['Volume'].iloc[-2])/hist['Volume'].iloc[-2]*100):.2f}%"
                )

            with col3:
                st.metric(
                    "52 Week High",
                    f"₹{hist['High'].max():.2f}"
                )

            # Stock Price Chart
            st.subheader("Stock Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='OHLC'
            ))
            fig.update_layout(
                template='plotly_dark',
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Technical Analysis
            st.subheader("Technical Analysis")
            tech_indicators = calculate_technical_indicators(hist)

            # Display technical indicators in columns
            tcol1, tcol2, tcol3 = st.columns(3)
            with tcol1:
                st.metric("RSI", f"{tech_indicators['RSI']:.2f}")
            with tcol2:
                st.metric("MACD", f"{tech_indicators['MACD']:.2f}")
            with tcol3:
                st.metric("Signal Line", f"{tech_indicators['Signal_Line']:.2f}")

            # Price Prediction Section
            st.subheader("Price Prediction (Next 7 Trading Days)")

            if len(hist) < 30:  # Check minimum required data
                st.warning("Insufficient historical data for prediction. Please select a longer time period.")
                return

            try:
                with st.spinner('Generating predictions...'):
                    predictions = predict_stock_price(hist)

                    if not predictions.empty:
                        # Create figure with both historical and predicted data
                        pred_fig = go.Figure()

                        # Add historical data (last 30 days)
                        pred_fig.add_trace(go.Scatter(
                            x=hist.index[-30:],
                            y=hist['Close'][-30:],
                            mode='lines',
                            name='Historical',
                            line=dict(color='#1f77b4')
                        ))

                        # Add predictions
                        pred_fig.add_trace(go.Scatter(
                            x=predictions.index,
                            y=predictions['Predicted'],
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='#2ca02c', dash='dash'),
                            marker=dict(size=8)
                        ))

                        pred_fig.update_layout(
                            template='plotly_dark',
                            title='Stock Price Prediction',
                            xaxis_title='Date',
                            yaxis_title='Price (₹)',
                            hovermode='x unified'
                        )

                        # Display prediction chart
                        st.plotly_chart(pred_fig, use_container_width=True)

                        # Display prediction values
                        st.write("Predicted Values:")
                        st.dataframe(predictions.round(2))

                        # Display prediction disclaimer
                        st.info("""
                            ℹ️ Prediction Disclaimer:
                            - Predictions are based on historical data and technical analysis
                            - Market conditions can change rapidly
                            - Use these predictions as one of many tools for analysis
                        """)
                    else:
                        st.warning("Unable to generate predictions. Please try with a different stock or time period.")

            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                st.warning("Unable to generate predictions. Please try with a different stock or time period.")

            # News Sentiment
            st.subheader("Recent News & Sentiment")
            news_data = get_news_sentiment(ticker.split('.')[0])

            for news in news_data:
                with st.expander(news['title']):
                    st.write(news['description'])
                    st.write(f"Sentiment: {news['sentiment']}")
                    st.write(f"Source: {news['source']}")

        except ValueError:
            st.error("The application has reached the API's rate limit. Please try again in a few minutes.")
            st.info("Tip: Avoid making repeated requests for the same stock. If the issue persists, try a different stock or time period.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
