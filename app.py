import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Clean data
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None


def create_features(data):
    """Create features from stock data"""
    df = data.copy()

    # Moving averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()

    # Lag features
    df['Close_Lag_1'] = df['Close'].shift(1)
    df['Close_Lag_3'] = df['Close'].shift(3)
    df['Close_Lag_7'] = df['Close'].shift(7)

    # Price changes
    df['Price_Change'] = df['Close'].pct_change() * 100
    df['Price_Momentum_3'] = ((df['Close'] - df['Close'].shift(3)) / df['Close'].shift(3)) * 100

    # Volume features
    df['Volume_Change'] = df['Volume'].pct_change() * 100
    df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_7']

    # Volatility
    df['Daily_Volatility'] = df['High'] - df['Low']
    df['ATR_7'] = df['Daily_Volatility'].rolling(window=7).mean()

    # Additional
    df['Gap'] = df['Open'] - df['Close'].shift(1)
    df['Gap_Percent'] = (df['Gap'] / df['Close'].shift(1)) * 100

    # Target
    df['Target'] = df['Close'].shift(-1)

    # Drop missing values
    df = df.dropna()

    return df


def train_model(df):
    """Train Linear Regression model"""
    # Select features
    feature_columns = [
        'Close_Lag_1', 'MA_7', 'MA_30', 'Price_Change',
        'Volume_Change', 'Daily_Volatility', 'Volume_Ratio',
        'Price_Momentum_3', 'ATR_7', 'Gap_Percent'
    ]

    X = df[feature_columns]
    y = df['Target']

    # Split data (80/20)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred)
    }

    return model, X_train, X_test, y_train, y_test, test_pred, metrics, feature_columns

# Title
st.title(" Stock Price Prediction Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙ Settings")

# Stock selection
ticker = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM"]
)

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=730)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now()
    )

# Run button
run_analysis = st.sidebar.button(" Run Analysis", type="primary")

# Info section
st.sidebar.markdown("---")
st.sidebar.markdown("""
###  About
This app predicts stock prices using:
- Linear Regression model
- Technical indicators
- Historical price data
""")

if run_analysis:
    with st.spinner(f"Downloading {ticker} data..."):
        data = download_stock_data(ticker, start_date, end_date)

    if data is not None and len(data) > 0:
        st.success(f" Downloaded {len(data)} days of data")

        # Create features
        with st.spinner("Creating features..."):
            df_featured = create_features(data)

        st.success(f" Created features ({len(df_featured)} samples ready)")

        # Train model
        with st.spinner("Training model..."):
            model, X_train, X_test, y_train, y_test, predictions, metrics, features = train_model(df_featured)

        st.success(" Model trained successfully!")

        st.markdown("##  Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Current Price",
                value=f"${data['Close'].iloc[-1]:.2f}"
            )

        with col2:
            st.metric(
                label="Test RMSE",
                value=f"${metrics['test_rmse']:.2f}"
            )

        with col3:
            st.metric(
                label="Test MAE",
                value=f"${metrics['test_mae']:.2f}"
            )

        with col4:
            st.metric(
                label="R² Score",
                value=f"{metrics['test_r2']:.4f}"
            )

        st.markdown("---")

        st.markdown("##  Predictions vs Actual")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test.index, y_test.values, label='Actual Price',
                linewidth=2, marker='o', markersize=4, color='blue')
        ax.plot(y_test.index, predictions, label='Predicted Price',
                linewidth=2, linestyle='--', marker='s', markersize=4, color='red')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(f'{ticker} - Model Predictions', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        st.markdown("##  Historical Price")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], linewidth=2, color='blue')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(f'{ticker} - Historical Closing Price', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)


        st.markdown("##  Feature Importance")

        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        })
        feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
        feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
        ax.barh(range(len(feature_importance)), feature_importance['Coefficient'],
                color=colors, alpha=0.7)
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance['Feature'])
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')

        st.pyplot(fig)

        st.markdown("##  Tomorrow's Prediction")

        # Get last features
        X_full = df_featured[features]
        last_features = X_full.iloc[-1:].values
        tomorrow_pred = model.predict(last_features)[0]

        current_price = float(data['Close'].iloc[-1])
        change = tomorrow_pred - current_price
        change_percent = (change / current_price) * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Today's Price",
                value=f"${current_price:.2f}"
            )

        with col2:
            st.metric(
                label="Predicted Tomorrow",
                value=f"${tomorrow_pred:.2f}",
                delta=f"${change:.2f}"
            )

        with col3:
            st.metric(
                label="Expected Change",
                value=f"{change_percent:.2f}%",
                delta=f"{'📈 UP' if change > 0 else '📉 DOWN'}"
            )

        st.markdown("##  Recent Predictions")

        results_df = pd.DataFrame({
            'Date': y_test.index[-10:],
            'Actual': y_test.values[-10:],
            'Predicted': predictions[-10:],
            'Error': (y_test.values[-10:] - predictions[-10:])
        })
        results_df['Actual'] = results_df['Actual'].map('${:.2f}'.format)
        results_df['Predicted'] = results_df['Predicted'].map('${:.2f}'.format)
        results_df['Error'] = results_df['Error'].map('${:.2f}'.format)

        st.dataframe(results_df, use_container_width=True)


        # Prepare full results
        full_results = pd.DataFrame({
            'Date': y_test.index,
            'Actual': y_test.values,
            'Predicted': predictions,
            'Error': y_test.values - predictions
        })

        csv = full_results.to_csv(index=False)


else:
    # Instructions when app first loads
    st.info(" Configure settings in the sidebar and click 'Run Analysis' to begin!")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit</p>
    <p>Stock Price Prediction Project</p>
</div>
""", unsafe_allow_html=True)