import yfinance as yf
import numpy as np

def calculate_var(stock_symbol, initial_investment, confidence_level=0.95, holding_period=1):
    # Download historical stock data
    stock_data = yf.download(stock_symbol, start="2021-01-01", end="2023-12-01")

    # Calculate daily returns
    stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()

    # Calculate mean and standard deviation of daily returns
    mean_return = stock_data['Daily_Return'].mean()
    std_dev = stock_data['Daily_Return'].std()

    # Calculate VaR assuming normal distribution
    z_score = np.percentile(np.random.normal(0, 1, 10000), confidence_level * 100)
    var = initial_investment * mean_return - z_score * (initial_investment * std_dev) * holding_period

    return var
  
    stock_symbol = "AAPL"  
    initial_investment = 100000 
    confidence_level = 0.95  # Confidence level for VaR calculation
    holding_period = 1  # Holding period in days

    calculated_var = calculate_var(stock_symbol, initial_investment, confidence_level, holding_period)
    
    print(f"VaR at {confidence_level * 100}% confidence level for {holding_period} day(s): ${calculated_var:.2f}")
