from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

class StockPricePredictor:
    """Stock price prediction model wrapper."""

    def __init__(self, model_type='linear'):
        """
        Initialize the predictor.

        Parameters:
        -----------
        model_type : str
            Type of model: 'linear' or 'knn'
        """
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'knn':
            self.model = KNeighborsRegressor(n_neighbors=5)
        else:
            raise ValueError("model_type must be 'linear' or 'knn'")

        self.model_type = model_type
        self.feature_names = None

    def prepare_data(self, data, feature_columns, target_column='Target', test_size=0.2):
        """
        Prepare data for training.

        Parameters:
        -----------
        data : pd.DataFrame
            Feature-engineered data
        feature_columns : list
            List of feature column names
        target_column : str
            Target column name
        test_size : float
            Proportion of data for testing

        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        X = data[feature_columns]
        y = data[target_column]

        self.feature_names = feature_columns

        # Use the last test_size portion of data for testing (time series)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        print(f"{self.model_type.capitalize()} model trained successfully!")

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)

        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'MAE': mean_absolute_error(y_test, predictions),
            'R2': r2_score(y_test, predictions)
        }

        return metrics, predictions

    def get_feature_importance(self):
        """Get feature importance (only for linear regression)."""
        if self.model_type == 'linear' and self.feature_names:
            importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': self.model.coef_
            })
            importance['Abs_Coefficient'] = abs(importance['Coefficient'])
            importance = importance.sort_values('Abs_Coefficient', ascending=False)
            return importance
        return None
