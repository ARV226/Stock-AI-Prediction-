import pandas as pd
import numpy as np

def calculate_technical_indicators(data):
    """Calculate technical indicators for the stock data."""
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    
    return {
        'RSI': rsi.iloc[-1],
        'MACD': macd.iloc[-1],
        'Signal_Line': signal_line.iloc[-1]
    }

def calculate_moving_averages(data):
    """Calculate moving averages for different periods."""
    ma_periods = [20, 50, 200]
    mas = {}
    
    for period in ma_periods:
        mas[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
    
    return mas
