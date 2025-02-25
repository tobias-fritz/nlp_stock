import torch
import pandas as pd
import numpy as np
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

    def __init__(self, data: list):
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

    def visualize_datset(self):
        '''Visualize the correlation between the inputs and the target.'''
        import matplotlib.pyplot as plt
        inputs = np.array([data['inputs'] for data in self.data])
        target = np.array([data['target'] for data in self.data])
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(inputs[:, 0, 0], target, 'o', label='Company')
        axs[0].plot(inputs[:, 0, 1], target, 'o', label='Product')
        axs[0].set_title('Positive Ratios vs Close Price')
        axs[0].set_xlabel('Positive Ratios')
        axs[0].set_ylabel('Close Price')
        axs[0].legend()
        axs[1].plot(inputs[:, 1, 0], target, 'o', label='Company')
        axs[1].plot(inputs[:, 1, 1], target, 'o', label='Product')

        axs[1].set_title('Positive Ratios vs Close Price')
        axs[1].set_xlabel('Positive Ratios')
        axs[1].set_ylabel('Close Price')
        axs[1].legend()

        plt.show()

def create_sentiment_dataset(result_df: pd.DataFrame) -> list:
    '''Create a SentimentDataset from the result DataFrame.

    Return a dataset of the news articles and their sentiments of the last 
    month built from the result DataFrame. We will roll over the data so that
    the inputs are sentiments of day 1, 2, and 3 and the target is the close of day 4.
    The next data point will be the sentiments of day 2, 3, and 4 and the target 
    is the close of day 5, and so on.

    The shape of the frame should be like here and compatible with the SentimentDataset:
    
    data = [
        {
            'inputs': [
                [positive_ratios_company[0], positive_ratios_product[0]],
                [positive_ratios_company[1], positive_ratios_product[1]],
                [positive_ratios_company[2], positive_ratios_product[2]]
            ],
            'target': close_day_4
        },
        {
            'inputs': [
                [positive_ratios_company[0], positive_ratios_product[0]],
                [positive_ratios_company[1], positive_ratios_product[1]],
                [positive_ratios_company[2], positive_ratios_product[2]]
            ],
            'target': close_day_4
        }
        ...
    ]

    Args:
    result_df (pd.DataFrame): The result DataFrame containing the processed sentiment data.

    Returns:
    list: A list of dictionaries containing the inputs and target for each data point
    
    '''


    if result_df.empty:
        raise ValueError("The result DataFrame is empty.")
    
    all_results_list = []
    for i in range(2, len(result_df)-1):
        result_slice = result_df[i-2:i+1]
        close_day_4 = result_df.iloc[i+1]['Close']
        positive_ratios_1 = result_slice['positive_ratio_1'].values
        positive_ratios_2 = result_slice['positive_ratio_2'].values
        
        day1 = [positive_ratios_1[0], positive_ratios_2[0]]
        day2 = [positive_ratios_1[1], positive_ratios_2[1]]
        day3 = [positive_ratios_1[2], positive_ratios_2[2]]

        all_results_list.append({
            'inputs': [day1, day2, day3],
            'target': close_day_4
        })

    return all_results_list
