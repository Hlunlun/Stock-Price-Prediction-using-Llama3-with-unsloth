import os
import requests
from utils import *

class StockAPI:
    def __init__(self, account, password):
        self.account = account
        self.password = password
        self.base_url = os.getenv('BASE_URL')
    
    def get_infos(self, stock_code, start_date, stop_date):
        info_url = f'{self.base_url}/api_get_stock_info_from_date_json/{str(stock_code)}/{str(start_date)}/{str(stop_date)}'
        results = requests.get(info_url).json()
        if(results['result'] == 'success'):
            for result in results['data']:
                result['date'] = convert_unix_to_date(result['date']) 
            return results['data']
        return dict([])
    
    def get_user_stocks(self):
        data = {'account': self.account, 'password':self.password }
        search_url = f'{self.base_url}/get_user_stocks'
        results = requests.post(search_url, data=data).json()
        if(results['result'] == 'success'):
            for result in results['data']:
                result['data'] = convert_unix_to_date(result['data']) 
            return results['data']
        return dict([])
    
    def buy_stock(self, stock_code, stock_shares, stock_price):
        
        data = {'account': self.account,
                'password': self.password,
                'stock_code': stock_code,
                'stock_shares': stock_shares,
                'stock_price': stock_price}
        buy_url = f'{self.base_url}/buy'
        result = requests.post(buy_url, data=data).json()        
        return result['result'] == 'success'
    
    def sell_stock(self, stock_code, stock_shares, stock_price):
        data = {'account': self.account,
                'password': self.password,
                'stock_code': stock_code,
                'stock_shares': stock_shares,
                'stock_price': stock_price}
        
        sell_url = f'{self.base_url}/sell'
        result = requests.post(sell_url, data=data).json()
        return result['result'] == 'success'
    
    
if __name__ == '__main__':
    api = StockAPI(os.getenv('ACCOUNT'), os.getenv('PASSWORD'))
    stock_info = api.get_infos('2330', '20220101', '20221231')