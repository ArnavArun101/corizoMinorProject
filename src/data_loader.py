import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        print(f"Successfully downloaded {len(data)} rows of data for {ticker}")
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def save_data(data, filepath):
    """Save data to CSV file."""
    data.to_csv(filepath)
    print(f"Data saved to {filepath}")

def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath, index_col=0, parse_dates=True)
