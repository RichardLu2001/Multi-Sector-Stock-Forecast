import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Function to download stock data and return a DataFrame
def download_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data['Adj Close']

# Function to plot ACF and PACF
def plot_acf_pacf(series, lags=40):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    # ACF plot
    plot_acf(series, lags=lags, ax=ax[0])
    ax[0].set_title('Autocorrelation Function (ACF)')

    # PACF plot
    plot_pacf(series, lags=lags, ax=ax[1])
    ax[1].set_title('Partial Autocorrelation Function (PACF)')

    plt.show()

# Function to perform Augmented Dickey-Fuller test for stationarity
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

stock_symbol = 'AAPL'
start_date = '2021-01-01'
end_date = '2023-12-01'

# Download stock data
stock_data = download_stock_data(stock_symbol, start_date, end_date)

# Plot stock prices
stock_data.plot(figsize=(12, 6), title=f'{stock_symbol} Stock Prices')
plt.show()

adf_test(stock_data)

diff_stock_data = stock_data.diff().dropna()

# Plot differenced series
diff_stock_data.plot(figsize=(12, 6), title=f'Differenced {stock_symbol} Stock Prices')
plt.show()

adf_test(diff_stock_data)

plot_acf_pacf(diff_stock_data, lags=40)

seasonal_difference = stock_data.diff(30).dropna()

result_adf = adfuller(seasonal_difference)
print(f'ADF Statistic: {result_adf[0]}')
print(f'p-value: {result_adf[1]}')
print(f'Critical Values: {result_adf[4]}')

plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_pacf(seasonal_difference, lags=20, title='Partial Autocorrelation Function (PACF)')
plt.subplot(212)
plot_acf(seasonal_difference, lags=20, title='Autocorrelation Function (ACF)')