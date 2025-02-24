import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader



class SentimentDataset(Dataset):
    ''' Sentiment dataset for the stock price prediction model.

    Example data structure:
    data = [
        {
            'inputs': [
                [positive_ratios_company[0][0], positive_ratios_product[0][0]],
                [positive_ratios_company[0][1], positive_ratios_product[0][1]],
                [positive_ratios_company[0][2], positive_ratios_product[0][2]]
            ],
            'target': close_day_4[0]
        },
        {
            'inputs': [
                [positive_ratios_company[1][0], positive_ratios_product[1][0]],
                [positive_ratios_company[1][1], positive_ratios_product[1][1]],
                [positive_ratios_company[1][2], positive_ratios_product[1][2]]
            ],
            'target': close_day_4[1]
        }
    ]
    '''

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.data[idx]['inputs'], dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(self.data[idx]['target'], dtype=torch.float32)
        return inputs, target
    
    def save(self, file_path):
        torch.save(self, file_path)
    
    @staticmethod
    def load(file_path):
        return torch.load(file_path)

def create_sentiment_dataset(result_df: pd.DataFrame) -> SentimentDataset:
    '''Create a SentimentDataset from the result DataFrame.'''

    raise NotImplementedError("Implement this function")

    data = [
        {
            'inputs': [
                [row['positive_ratio_1'], row['positive_ratio_2']],
                [row['article_count_1'], row['article_count_2']],
            ],
            'target': row['Close']
        }
        for _, row in result_df.iterrows()
    ]
    return SentimentDataset(data)