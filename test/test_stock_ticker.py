import pytest
import pandas as pd
from datetime import datetime
from src.stock_ticker import get_stock_close

def test_get_stock_close_has_correct_columns():
    """Test that the DataFrame has the expected columns"""
    df = get_stock_close("MSFT")
    assert list(df.columns) == ['Date', 'Close']
    assert df['Date'].dtype == 'datetime64[ns]'
    assert df['Close'].dtype == 'float64'

def test_get_stock_close_data_range():
    """Test that the DataFrame contains approximately one month of data"""
    df = get_stock_close("GOOGL")
    days_difference = (df['Date'].max() - df['Date'].min()).days
    assert 28 <= days_difference <= 31  # Allowing for some flexibility in month length

def test_get_stock_close_invalid_ticker():
    """Test that the function raises an error for invalid ticker"""
    with pytest.raises(Exception):
        get_stock_close("INVALID_TICKER_SYMBOL")