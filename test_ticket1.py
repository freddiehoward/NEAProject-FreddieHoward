import pytest
import yfinance as yf

# Global setup
stock_ticker = "NVDA"
ticker_obj = yf.Ticker(stock_ticker)

def test_ticker_initialization():
    '''Test if the Ticker object is correctly initialized'''
    assert ticker_obj.ticker == stock_ticker

def test_fetch_historical_data_not_empty():
    '''Test that historical data is retrieved and the DataFrame is not empty'''
    df = ticker_obj.history(period="10y")
    assert len(df) > 0, "The historical data DataFrame should not be empty."

def test_columns_present_in_historical_data():
    '''Test that the historical data includes the essential columns'''
    df = ticker_obj.history(period="10y")
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    
    '''Check each column individually'''
    assert 'Open' in df.columns, "The 'Open' column is missing from the data."
    assert 'High' in df.columns, "The 'High' column is missing from the data."
    assert 'Low' in df.columns, "The 'Low' column is missing from the data."
    assert 'Close' in df.columns, "The 'Close' column is missing from the data."
    assert 'Volume' in df.columns, "The 'Volume' column is missing from the data."
    assert 'Dividends' in df.columns, "The 'Dividends' column is missing from the data."
    assert 'Stock Splits' in df.columns, "The 'Stock Splits' column is missing from the data."

def test_drop_dividends_and_splits():
    '''Test that 'Dividends' and 'Stock Splits' columns can be removed'''
    df = ticker_obj.history(period="10y")
    df.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
    
    # Check if the columns were removed by verifying the length of the columns
    assert 'Dividends' not in df.columns, "'Dividends' column should be removed."
    assert 'Stock Splits' not in df.columns, "'Stock Splits' column should be removed."

def test_close_data_for_prediction():
    '''Test that the 'Close' column data is available for prediction'''
    df = ticker_obj.history(period="10y")
    close_data = df['Close'].values
    
    # Ensure that there is data in the 'Close' column for prediction
    assert len(close_data) > 0, "'Close' column data should be available for predictions."

def test_volume_data_consistency():
    '''Test that 'Volume' column values are non-negative'''
    df = ticker_obj.history(period="10y")
    volume_data = df['Volume'].values

    # Ensure each volume value is non-negative
    for volume in volume_data:
        assert volume >= 0, "Volume data should be non-negative."
