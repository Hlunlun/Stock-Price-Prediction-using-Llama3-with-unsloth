import os
import pandas as pd
from utils import *
from torch.utils.data import Dataset
# from newsapi import NewsApiClient

class StockDataset(Dataset):
    def __init__(self, data_dir=None, df=None, fields=None):
        self.data = []        
        self.context = {
                'role': 'system',
                'content': "You are a stock trader. You are given a company name and a list of dates. You need to predict the next n days price of the company."
            }
        
        if fields is None:
            self.fields = ['date', 'open', 'high', 'low', 'close', 'change', 'transaction']

        else:
            self.fields = fields
        

        if data_dir is not None:
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                company = file.split('_')[0]
                df = pd.read_csv(file_path)
                self.data += self.organize_data(df, company)
        elif df is not None:
            company = df['company'].iloc[0]
            self.data = self.organize_data(df, company)

        save_to_json(self.data, 'all_stock_data.json')

    def organize_data(self, df, company):
        data = []
        data_len = len(df)

        # Define prediction configurations: (history_length, prediction_length, min_required_length)
        configs = [
            (1, 1, 2),    # 1-day prediction (requires >1 day)
            (5, 5, 10),    # 5-day prediction (requires >5 days)
            (30, 5, 35),  # 1-month history, 5-day prediction
            (90, 30, 120), # 3-month history, 1-month prediction
        ]

        for history, predict, min_len in configs:
            if data_len >= min_len:
                prompt_data = self.create_prompt(df, history, predict, company)
                data.extend(prompt_data)

        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def create_prompt(self, df, day, pred_day, company):
        stock_nd_data = []
        for l in range(len(df) - day - pred_day + 1):
            df_input = df.iloc[l:l+day]
            df_target = df.iloc[l+day:l+day+pred_day]
            
            input_prompt = "Here is past {} day stock price of {} company.\n\n".format(day, company)
            for _, row in df_input.iterrows():
                for field in self.fields:
                    if field in row:
                        input_prompt += f'{field}: {row[field]}, '
                input_prompt += '\n'                
            input_prompt += f'Predict the next {pred_day} day prices:\n\n'
            
            target_prompt = "The next {} day prices are:\n".format(pred_day)
            for _, row in df_target.iterrows():
                for field in self.fields:
                    if field in row:
                        target_prompt += f'{field}: {row[field]}, '
                target_prompt += '\n'

            
            stock_nd_data.append([
                self.context,
                {'content': input_prompt, 'role': 'user'},
                {'content': target_prompt, 'role': 'assistant'}
            ])
        
        return stock_nd_data
        
   

class NewsDataset(Dataset):
    def __init__(self):
        self.data = []




