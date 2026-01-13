# Stock Price Prediction Project

## Overview
This project implements a stock price prediction system using machine learning techniques (Linear Regression and KNN).
It includes an interactive web dashboard built with Streamlit for visualizing predictions and model performance.

## Project Structure
```
stock_prediction/
│
├── data/                          # Downloaded stock data storage
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/                           # Source code modules
│   ├── data_loader.py            # Data loading functions
│   ├── feature_engineer.py       # Feature engineering functions
│   └── model.py                  # ML model implementation
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone or download this project
2. Navigate to the project directory
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App
```bash
streamlit run app.py
```

### Using Jupyter Notebooks
```bash
jupyter notebook
```

Then open the notebooks in the `notebooks/` folder in sequence.

### Using the Modules Directly
```python
from src.data_loader import load_stock_data
from src.feature_engineer import engineer_features
from src.model import StockPricePredictor

# Load data
data = load_stock_data('AAPL', '2022-01-01', '2024-01-01')

# Engineer features
featured_data = engineer_features(data)

# Train model
predictor = StockPricePredictor(model_type='linear')
# ... rest of the training code
```

## Features

- **Data Loading**: Download historical stock data from Yahoo Finance
- **Feature Engineering**: Create technical indicators (MA, volatility, lag features)
- **Model Training**: Train Linear Regression or KNN models
- **Evaluation**: Calculate RMSE, MAE, and R² metrics
- **Visualization**: Interactive charts and dashboards
- **Web Interface**: User-friendly Streamlit application

## Disclaimer

⚠️ **This is an educational project only!**

- Do not use these predictions for actual trading decisions
- Stock markets are influenced by many unpredictable factors
- Historical performance does not guarantee future results
- Always consult with financial advisors before making investment decisions

## Learning Objectives

- Apply machine learning to real-world financial data
- Practice data preprocessing and feature engineering
- Understand model evaluation metrics
- Build interactive web applications
- Work with time series data

## Extensions

Possible improvements to this project:
- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Implement ensemble methods
- Add prediction confidence intervals
- Create a trading strategy simulator
- Support for multiple stock comparisons

## License

This project is for educational purposes only.
