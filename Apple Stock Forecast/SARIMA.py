from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA


# Function to split data into training and testing sets
def train_test_split(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train, test = data[:train_size], data[train_size:]
    return train, test

# Function to fit SARIMA model and make predictions
def fit_arima(train_data, test_data, order,seasonal_order):
    model = ARIMA(train_data, order=order,seasonal_order=seasonal_order)
    fitted_model = model.fit()
    predictions = fitted_model.forecast(steps=len(test_data))
    return predictions

# Function to calculate Root Mean Squared Error (RMSE)
def calculate_rmse(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    return rmse

# Read data from CSV file
file_path = 'AAPL.csv'
stock_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)

# Convert the index to datetime format
stock_data.index = pd.to_datetime(stock_data.index, errors='coerce')


# Plot stock prices
stock_data['Adj Close'].plot(figsize=(12, 6), title='Stock Prices')
plt.show()

# Split data into training and testing sets
train_data, test_data = train_test_split(stock_data['Adj Close'], train_ratio=0.9)

# Fit SARIMA model
order = (2, 1, 2)  # Example order (p, d, q)
seasonal_order = (1,1,0,30)
predictions = fit_arima(train_data, test_data,order,seasonal_order)

# Plot actual vs. predicted prices
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label='Training Data')
plt.plot(test_data.index, test_data, label='Actual Test Data')
plt.plot(test_data.index, predictions, label='Predicted Test Data', linestyle='dashed')
plt.title('Stock Prices - ARIMA Predictions')
plt.legend()
plt.show()


# Calculate RMSE, MAPE, MAE and R^2
rmse = calculate_rmse(test_data, predictions)
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

mape = mean_absolute_error(test_data, predictions) / np.mean(test_data) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

mae = mean_absolute_error(test_data, predictions)
print(f'Mean Absolute Error (MAE): {mae}')

r_squared = r2_score(test_data, predictions)
print(f'R-squared: {r_squared}')
