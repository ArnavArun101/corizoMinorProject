import pandas as pd
import numpy as np

def calculate_moving_average(data, column='Close', window=7):
    return data[column].rolling(window=window).mean()

def calculate_price_change(data, column='Close'):
    """Calculate percentage change from previous day."""
    return data[column].pct_change() * 100

def create_lag_features(data, column='Close', lags=[1, 3, 7]):
    df = data.copy()
    for lag in lags:
        df[f'{column}_Lag_{lag}'] = df[column].shift(lag)
    return df

def engineer_features(data):
    df = data.copy()

    # Moving averages
    df['MA_7'] = calculate_moving_average(df, window=7)
    df['MA_30'] = calculate_moving_average(df, window=30)

    # Price changes
    df['Price_Change'] = calculate_price_change(df, column='Close')
    df['Volume_Change'] = calculate_price_change(df, column='Volume')

    # Volatility (High-Low difference)
    df['Volatility'] = df['High'] - df['Low']

    # Lag features
    df = create_lag_features(df, column='Close', lags=[1, 3, 7])

    # Target variable (next day's close price)
    df['Target'] = df['Close'].shift(-1)

    # Drop rows with missing values
    df = df.dropna()

    return df
