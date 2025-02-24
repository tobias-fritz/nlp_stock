import yfinance as yf
import pandas as pd


def get_stock_close(ticker: str) -> pd.DataFrame:
    '''Get the stock close prices for the last month.'''
    return (yf.Ticker(ticker)
            .history(period="1mo")
            .reset_index()
            [['Date', 'Close']]
            .assign(Date=lambda x: pd.to_datetime(x['Date']).dt.tz_localize(None)))