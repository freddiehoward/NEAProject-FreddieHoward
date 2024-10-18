import pytest
import yfinance as yf

# Global setup
stock_ticker = "NVDA"
ticker_obj = yf.Ticker(stock_ticker)

def test_ticker_initialization():
    '''Test if the Ticker object is correctly initialized'''
    assert ticker_obj.ticker == stock_ticker




