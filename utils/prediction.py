import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def prepare_data(data, lookback=30):
    """Prepare data for prediction model with improved error handling."""
    try:
        if data is None or len(data) < lookback:
            raise ValueError(f"Insufficient data. Need at least {lookback} days of historical data.")

        # Create features dataframe
        features = pd.DataFrame()
        features['Close'] = data['Close']

        # Add technical indicators
        features['MA5'] = features['Close'].rolling(window=5).mean()
        features['MA20'] = features['Close'].rolling(window=20).mean()

        # Handle missing values
        features = features.ffill().bfill()

        if features.isnull().any().any():
            raise ValueError("Unable to handle missing values in the dataset")

        # Scale the features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features)

        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])  # Predict next day's closing price

        return np.array(X), np.array(y), scaler, features.columns

    except Exception as e:
        raise ValueError(f"Error preparing data: {str(e)}")

def predict_stock_price(data):
    """Predict stock prices for the next 7 days with improved model and error handling."""
    try:
        # Prepare data
        X, y, scaler, feature_columns = prepare_data(data)

        if len(X) == 0:
            raise ValueError("No valid data available for prediction")

        # Reshape input for RandomForest
        X_reshaped = X.reshape(X.shape[0], -1)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_reshaped, y)

        # Prepare last sequence for prediction
        last_sequence = X[-1]

        # Generate future dates excluding weekends
        future_dates = []
        current_date = data.index[-1]
        predictions = []

        for _ in range(7):
            current_date += timedelta(days=1)
            while current_date.weekday() > 4:  # Skip weekends
                current_date += timedelta(days=1)
            future_dates.append(current_date)

            # Make prediction
            current_sequence = last_sequence.reshape(1, -1)
            pred = model.predict(current_sequence)
            scaled_pred = pred[0]

            # Transform prediction back to original scale
            orig_shape_pred = np.zeros((1, len(feature_columns)))
            orig_shape_pred[0, 0] = scaled_pred
            actual_pred = scaler.inverse_transform(orig_shape_pred)[0, 0]
            predictions.append(actual_pred)

            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = scaled_pred

        # Create prediction DataFrame
        pred_df = pd.DataFrame(
            predictions,
            index=future_dates,
            columns=['Predicted']
        )

        return pred_df

    except Exception as e:
        print(f"Prediction error: {str(e)}")  # For debugging
        return pd.DataFrame(columns=['Predicted'])  # Return empty DataFrame on error