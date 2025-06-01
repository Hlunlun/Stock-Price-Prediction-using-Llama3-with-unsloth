import os
import re
import json
import requests
import pandas as pd
# from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from twstock import Stock

load_dotenv()


# def collect_platform_rules():
#     rule_ids = ["price-rule", "fee-rule", "performance-rule", "faq-rule"]
#     "Collect price rule, fee rule, performance, and faq rule"
#     url = "https://ciot.imis.ncku.edu.tw/stock/trading_regulation/"
#     response = requests.get(url)
    
#     if response.status_code != 200:
#         print("Failed to retrieve the webpage.")
#         return None
    
#     soup = BeautifulSoup(response.text, "html.parser")

#     rules = {}    
#     for rule_id in rule_ids:
#         rule_section = soup.find(id=rule_id)
#         if rule_section:
#             rules[rule_id] = rule_section.get_text(strip=True, separator="\n")
#         else:
#             rules[rule_id] = "Not found"
    
#     return rules

def convert_unix_to_date(unix_timestamp):
    """Convert a Unix timestamp to a date string in 'YYYY-MM-DD' format."""
    try:
        timestamp = int(float(unix_timestamp))
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        raise ValueError(f"Invalid Unix timestamp: {unix_timestamp}")

def save_to_json(data, filename, dir='./'):
    os.makedirs(dir, exist_ok=True)
    file_path = os.path.join(dir, filename)
    with open(file_path, 'w',encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json(filename, dir='./'):
    file_path = os.path.join(dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_workdays(start_date, n_days, direction='past'):
    current_date = start_date
    delta = timedelta(days=-1 if direction == 'past' else 1)
    workday_count = 0
    
    # Check if start date is a workday
    if current_date.weekday() < 5:
        workday_count = 1
        if workday_count == n_days:
            return current_date
    
    while workday_count < n_days:
        current_date += delta
        # Monday=0, Sunday=6, so weekdays are 0-4
        if current_date.weekday() < 5:
            workday_count += 1
    
    return current_date

def fetch_twse_data(code, date, count):
    # twse_url = os.getenv('TWSE_URL')
    # url = twse_url.format(y=date.year, m=date.month, d=date.day, code=code)
    # response = requests.get(url)
    # if response.status_code != 200:
    #     return None
    # json_data = response.json()
    columns = ['date', 'capacity', 'turnover', 'open', 'high', 'low', 'close', 'change', 'transaction']
    # float_fields = ['capacity', 'turnover', 'open', 'high', 'low', 'close', 'change', 'transaction']
    # target_date = date.strftime('%Y/%m/%d')
    target_date = date.strftime('%Y-%m-%d')

    stock = Stock(code)
    json_data = stock.fetch_from(date.year, date.month)

    result = []

    # for row in json_data['data']:
    for row in json_data:
        row_dict = dict(zip(columns, row))
        row_dict['date'] = row_dict['date'].strftime('%Y-%m-%d')

        # 民國年轉西元年
        row_dict['date'] = re.sub(
            r'(\d+)(/\d+/\d+)',
            lambda m: str(int(m.group(1)) + 1911) + m.group(2),
            row_dict['date']
        )

        # 數值欄位轉成 float（移除逗號、處理 + / 空白）
        # for field in float_fields:
        #     try:
        #         cleaned = row_dict[field].replace(',', '').replace('+', '').strip()
        #         row_dict[field] = float(cleaned)
        #     except (ValueError, AttributeError):
        #         row_dict[field] = float('nan')

        if row_dict['date'] >= target_date:
            result.append(row_dict)
    
    return result if result else None

def search_company_name_code(code=None, name=None):
    url = os.getenv('COMPANY_LIST_URL')
    response = requests.get(url)
    if response.status_code == 200:
        company_list = response.json()
        for company in company_list:
            if code is not None and company['公司代號'] == code:
                return company['公司名稱']
            elif name is not None and company['公司名稱'] == name:
                return company['公司代號']
        return None
    else:
        return None


import re

def parse_prediction_output(text, fields):
    results = []

    patterns = {
        # 'date': r'date:\s*(\d{4}/\d{2}/\d{2})', 
        'date': r'date:\s*(\d{4}-\d{2}-\d{2})',
        'capacity': r'capacity:\s*([\d,]+)',
        'turnover': r'turnover:\s*([\d,]+)',
        'open': r'open:\s*(\d+\.\d+)',
        'high': r'high:\s*(\d+\.\d+)',
        'low': r'low:\s*(\d+\.\d+)',
        'close': r'close:\s*(\d+\.\d+)',
        'change': r'change:\s*(-?\d+\.\d+)',
        'transaction': r'transaction:\s*([\d,]+)'
    }
    
    try:
        pattern = r',\s*'.join([patterns[f] for f in fields])
    except KeyError as e:
        raise ValueError(f"Unsupported field: {e}")

    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    for match in matches:
        record = {}
        for i, field in enumerate(fields):
            value = match[i]
            if field == 'date':
                record[field] = value
            else:
                record[field] = float(value.replace(',', ''))
        results.append(record)

    return results
