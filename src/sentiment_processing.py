
from typing import List, Dict
import pandas as pd
import datetime


def process_sentiment(sentiment_1: List[str], 
                      sentiment_2: List[str],
                      closing_prices: pd.DataFrame, 
                      window_size: int
                      ) -> Dict[str, int]:
    '''Process the sentiment data and return the positive ratio.

    Herein use the sentiments of the COmpany and their product class 
    to determine the positive ratio for each, in the end merge the two 
    dataframes with the closing prices of the stock and return the result.


    '''


    positive_ratios_1 = _get_positive_ratio(sentiment_1, window_size)
    positive_ratios_2 = _get_positive_ratio(sentiment_2, window_size)

    # merge the two dataframes
    positive_ratios = positive_ratios_1.merge(positive_ratios_2, left_index=True, right_index=True, how='outer')
    result = closing_prices.merge(positive_ratios, left_on='Date', right_index=True, how='left')

    # Min max normalize each columns
    result = result.assign(
        Close=_min_max_normalize(result['Close']),
        positive_ratio_1=_min_max_normalize(result['positive_ratio_1']),
        positive_ratio_2=_min_max_normalize(result['positive_ratio_2'])
    )

    return 


def _get_positive_ratio(sentiment: List[Dict[str, str]], 
                        window_size: int,
                        ) -> pd.DataFrame:
    '''Get the positive ratio of the sentiment data.'''

    sentiment_df = pd.DataFrame(sentiment)
    positive_ratios = (sentiment_df
        .groupby('date')
        .agg({
            'sentiment': lambda x: [
                (x == 'POSITIVE').mean(),
                len(x)
            ]
        })
        .apply(lambda x: x['sentiment'], axis=1)
        .to_dict())

    positive_ratios = (pd.DataFrame(positive_ratios).T
        .rename(columns={0: 'positive_ratio', 1: 'article_count'})
        .sort_index())
    # Convert positive_ratios index to datetime
    positive_ratios.index = pd.to_datetime(positive_ratios.index)

    rolling_average = (
        (positive_ratios['positive_ratio'] * positive_ratios['article_count'])
        .rolling(window=window_size, min_periods=1)
        .sum() / 
        positive_ratios['article_count']
        .rolling(window=window_size, min_periods=1)
        .sum()
    )

    # Convert positive_ratios index to datetime first, then make it timezone-naive
    rolling_average.index = pd.to_datetime(rolling_average.index).tz_localize(None)
    return pd.DataFrame(rolling_average, columns=['positive_ratio'])

def _min_max_normalize(series: pd.Series) -> pd.Series:
    '''Min-max normalize the input series to range [0,1].'''
    return (series - series.min()) / (series.max() - series.min())