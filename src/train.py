import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.dataset import SentimentDataset
from src.sentiment_processing import process_sentiment
from src.newsapi import get_last_month_stock_sentiment
from src.stock_ticker import get_stock_close
from src.dataset import create_sentiment_dataset, SentimentDataset
from src.model import SentimentModel
from torchmetrics import Accuracy



def train_model(classifier: callable, # the sentiment analysis pipeline
                stock_list: str, # the file that contains the stocks to extract the sentiments and build the dataset
                api_key: str, # the file containing the NewsAPI key
                window_size: int, # the window size for the dataset
                batch_size: int, # the batch size for the training
                n_epochs: int, # the number of epochs to train the model
                learning_rate: float=0.001, # the learning rate for the optimizer
                model_dir: str="model/", # the directory to save the model
                ):
    '''Train the sentiment analysis model on the news articles
    and save the model to the models directory.
    '''

    # 1. Load the stock list
    stock_list = pd.read_csv(stock_list, header=1, delimiter=',')

    # 2. Load the NewsAPI key
    with open(api_key, 'r') as f:
        api_key = f.read()
    
    # 3. Build a dataset of the news articles and their sentiments
    dataset = np.array(shape=(0, 3, 2))
    for index, row in stock_list.iterrows():
        company, stock_name, product_class = row
        processed_data = process_sentiment(get_last_month_stock_sentiment(company, classifier, api_key), 
                                           get_last_month_stock_sentiment(product_class, classifier, api_key), 
                                           get_stock_close(stock_name), 
                                           window_size)
        process_sentiment = create_sentiment_dataset(processed_data)
        dataset = np.concatenate((dataset, process_sentiment), axis=0)
    sentiment_dataset = SentimentDataset(dataset)

    # 4. Train test split 80, 20
    train_size, test_size = int(0.8 * len(sentiment_dataset)), int(0.2 * len(sentiment_dataset))
    train_data, test_data = torch.utils.data.random_split(sentiment_dataset, [train_size, test_size])

    # 5. Setup training with dataloader, model, criterion, optimizer
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = SentimentModel(2, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 6. Train the model
    model_tracker = {'train_loss': [], 
                     'train_accuracy': [],
                     'test_loss': None, 
                     'test_accuracy': None,
                     'epoch': [], 
                     'learning_rate': learning_rate}
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_accuracy = Accuracy(task='float')
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_accuracy(outputs, targets)
        
        model_tracker['train_loss'].append(epoch_loss)
        model_tracker['epoch'].append(epoch)
        model_tracker['train_accuracy'].append(epoch_accuracy.compute())

    # 7. Test the model
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_accuracy = Accuracy(task='float')
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            test_accuracy(outputs, targets)
        model_tracker['test_loss'] = test_loss
        model_tracker['test_accuracy'] = test_accuracy.compute()

    # 8. Save the model
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model, f"{model_dir}sentiment_model.pt")
    torch.save(model_tracker, f"{model_dir}model_tracker.pt")