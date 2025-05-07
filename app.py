import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import requests
from datetime import datetime, timedelta
from utils.stock_analysis import calculate_technical_indicators
from utils.news_sentiment import get_news_sentiment
from utils.prediction import predict_stock_price
from nsepython import nsefetch
import numpy as np

# Alpha Vantage config
API_KEY = "HTQ5XE346QAFLKUI"  # Replace with your real key
BASE_URL = "https://www.alphavantage.co/query"

# Streamlit config
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

@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    if '.BSE' in symbol.upper():
        raise ValueError("BSE symbols are not supported. Please use .NSE symbols.")
    symbol = symbol.replace(".NSE", "").upper()

    try:
        url = f"https://www.nseindia.com/api/chart-databyindex?index={symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br"
        }
        data = nsefetch(url)

        prices = data['grapthData']
        df = pd.DataFrame(prices, columns=["Timestamp", "Close"])
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df.set_index("Date", inplace=True)
        df["Close"] = df["Close"].astype(float)

        # Simulate OHLC for charting
        df["Open"] = df["Close"].shift(1).fillna(df["Close"])
        df["High"] = df[["Open", "Close"]].max(axis=1) * np.random.uniform(1.00, 1.02, len(df))
        df["Low"] = df[["Open", "Close"]].min(axis=1) * np.random.uniform(0.98, 1.00, len(df))
        df["Volume"] = np.random.randint(100000, 500000, len(df))
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to fetch NSE data: {str(e)}")

def main():
    st.title("📈 Stock Prediction & Analysis")
    
    st.sidebar.header("Settings")
    ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.BSE or TCS.NSE)", "RELIANCE.BSE")
    period = st.sidebar.selectbox("Select Time Period", ("1mo", "3mo", "6mo", "1y", "2y", "5y"))

    if st.sidebar.button("Analyze"):
        try:
            with st.spinner('Fetching stock data...'):
                hist = get_stock_data(ticker)

                # Filter date range
                days = {
                    "1mo": 30, "3mo": 90, "6mo": 180,
                    "1y": 365, "2y": 730, "5y": 1825
                }[period]
                hist = hist[hist.index >= datetime.today() - timedelta(days=days)]

                if hist.empty:
                    st.error("No data found for the specified ticker or period.")
                    return

                col1, col2, col3 = st.columns(3)

                with col1:
                    current_price = hist['Close'].iloc[-1]
                    price_change = current_price - hist['Close'].iloc[-2]
                    price_change_pct = (price_change / hist['Close'].iloc[-2]) * 100
                    st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change_pct:.2f}%")

                with col2:
                    st.metric("Volume", f"{hist['Volume'].iloc[-1]:,.0f}",
                        f"{((hist['Volume'].iloc[-1] - hist['Volume'].iloc[-2])/hist['Volume'].iloc[-2]*100):.2f}%")

                with col3:
                    st.metric("52 Week High", f"₹{hist['High'].max():.2f}")

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
                fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Technical Analysis")
                tech_indicators = calculate_technical_indicators(hist)

                tcol1, tcol2, tcol3 = st.columns(3)
                with tcol1:
                    st.metric("RSI", f"{tech_indicators['RSI']:.2f}")
                with tcol2:
                    st.metric("MACD", f"{tech_indicators['MACD']:.2f}")
                with tcol3:
                    st.metric("Signal Line", f"{tech_indicators['Signal_Line']:.2f}")

                st.subheader("Price Prediction (Next 7 Trading Days)")
                if len(hist) < 30:
                    st.warning("Insufficient historical data for prediction. Please select a longer time period.")
                    return

                try:
                    with st.spinner('Generating predictions...'):
                        predictions = predict_stock_price(hist)

                        if not predictions.empty:
                            pred_fig = go.Figure()
                            pred_fig.add_trace(go.Scatter(
                                x=hist.index[-30:],
                                y=hist['Close'][-30:],
                                mode='lines',
                                name='Historical',
                                line=dict(color='#1f77b4')
                            ))
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
                            st.plotly_chart(pred_fig, use_container_width=True)

                            st.write("Predicted Values:")
                            try:
                                styled_df = predictions.style.format("{:.2f}")
                                styled_df = styled_df.background_gradient(cmap='RdYlGn', axis=0)
                                st.dataframe(styled_df)
                            except Exception:
                                st.dataframe(predictions.round(2))

                            st.info("""
                                ℹ️ Prediction Disclaimer:
                                - Predictions are based on historical data and technical analysis
                                - Market conditions can change rapidly
                                - Use these predictions as one of many tools for analysis
                            """)
                        else:
                            st.warning("Unable to generate predictions. Try a different stock or longer time frame.")
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

                st.subheader("Recent News & Sentiment")
                news_data = get_news_sentiment(ticker.split('.')[0])
                for news in news_data:
                    with st.expander(news['title']):
                        st.write(news['description'])
                        st.write(f"Sentiment: {news['sentiment']}")
                        st.write(f"Source: {news['source']}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
