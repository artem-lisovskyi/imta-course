import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Function to fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to perform Fourier analysis
def fourier_analysis(data, column):
    close_prices = data[column].values
    fourier_transform = np.fft.fft(close_prices)
    frequencies = np.fft.fftfreq(len(fourier_transform))
    return frequencies, fourier_transform

def arima_analysis(data, column, order):
    model = ARIMA(data[column], order=order)
    results = model.fit()
    predictions = results.predict(start=0, end=len(data)-1)
    return predictions
# Function to filter out high-frequency noise
def filter_high_frequency(frequencies, fourier_transform, threshold):
    fourier_transform_filtered = fourier_transform.copy()
    fourier_transform_filtered[np.abs(frequencies) > threshold] = 0
    return fourier_transform_filtered

# Function to inverse Fourier transform and get predicted values
def inverse_fourier_transform(filtered_transform):
    predicted_values = np.fft.ifft(filtered_transform)
    return predicted_values.real

# Function to fit Exponential Smoothing model and get predictions
def exponential_smoothing_analysis(data, column, trend='add', seasonal='add', seasonal_periods=12):
    model = ExponentialSmoothing(data[column], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    results = model.fit()
    predictions = results.predict(start=0, end=len(data)-1)
    return predictions

# Function to calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
def evaluate_predictions(actual_values, predicted_values):
    mse = mean_squared_error(actual_values, predicted_values)
    mae = mean_absolute_error(actual_values, predicted_values)
    return mse, mae

# Function to plot actual and predicted values
def plot_predictions(data, actual_values, predicted_values, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, actual_values, label='Actual Prices', color='blue')
    plt.plot(data.index, predicted_values, label='Predicted Prices', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Set the stock symbol, start date, and end date
    stock_symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    # Get historical stock data
    stock_data = get_stock_data(stock_symbol, start_date, end_date)

    # Perform Fourier analysis
    frequencies, fourier_transform = fourier_analysis(stock_data, 'Close')

    # Filter out high-frequency noise
    threshold = 0.01  # Adjust the threshold based on the dataset
    filtered_transform = filter_high_frequency(frequencies, fourier_transform, threshold)

    # Inverse Fourier transform to get predicted values
    fourier_predicted_values = inverse_fourier_transform(filtered_transform)

    # Fit Exponential Smoothing model and get predictions
    exp_smoothing_predicted_values = exponential_smoothing_analysis(stock_data, 'Close')

     # Fit ARIMA model and get predictions
    arima_order = (5, 1, 0)
    arima_predicted_values = arima_analysis(stock_data, 'Close', arima_order)

    # Plot actual and predicted values for all methods
    plot_predictions(stock_data, stock_data['Close'], fourier_predicted_values, f'Stock Prediction using Fourier Analysis for {stock_symbol}')
    plot_predictions(stock_data, stock_data['Close'], arima_predicted_values, f'Stock Prediction using ARIMA for {stock_symbol}')
    plot_predictions(stock_data, stock_data['Close'], exp_smoothing_predicted_values, f'Stock Prediction using Exponential Smoothing for {stock_symbol}')

    # Evaluate predictions for all methods
    mse_fourier, mae_fourier = evaluate_predictions(stock_data['Close'].values, fourier_predicted_values)
    mse_exp_smoothing, mae_exp_smoothing = evaluate_predictions(stock_data['Close'].values, exp_smoothing_predicted_values)
    mse_arima, mae_arima = evaluate_predictions(stock_data['Close'].values, arima_predicted_values)

    print(f'Fourier Analysis - Mean Squared Error (MSE): {mse_fourier}, Mean Absolute Error (MAE): {mae_fourier}')
    print(f'Exponential Smoothing - Mean Squared Error (MSE): {mse_exp_smoothing}, Mean Absolute Error (MAE): {mae_exp_smoothing}')
    print(f'ARIMA Analysis - Mean Squared Error (MSE): {mse_arima}, Mean Absolute Error (MAE): {mae_arima}')