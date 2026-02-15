from fred_key import fred_key
import requests
import pandas as pd
import json
from datetime import datetime
from metrics_computing import TimeSeriesAnalyzer
#import matplotlib.pyplot as plt

# observations
obs_endpoint = 'series/observations'

# api key
api_key = fred_key
# url
base_url = 'https://api.stlouisfed.org/fred/'

def load_indicator_metadata():
    with open('output.json', 'r', encoding='utf-8') as f:
        indicators = json.load(f)
    return {item['SERIES']: item for item in indicators}

def call_fred_api(series_id, start_date, end_date):
    """call FRED API to get data"""
    indicators_map = load_indicator_metadata()
    frequency = None
    indicator_name = series_id
    
    if series_id in indicators_map:
        frequency = indicators_map[series_id]['PERIOD'].lower()
        indicator_name = indicators_map[series_id]['INDICATOR']
    
    params = {
        'series_id': series_id,
        'api_key': fred_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date,
    }
    
    if frequency:
        params['frequency'] = frequency
    
    response = requests.get(base_url + obs_endpoint, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'observations' in data and len(data['observations']) > 0:
            try:
                analyzer = TimeSeriesAnalyzer(data['observations'])
                analysis = analyzer.generate_summary()
                
                return {
                    'success': True,
                    'series_id': series_id,
                    'indicator_name': indicator_name,
                    'data': data['observations'],
                    'analysis': analysis  # 添加分析结果
                }
            except Exception as e:
                print(f"Warning: Analysis failed: {e}")
                # 即使分析失败，仍返回原始数据
                return {
                    'success': True,
                    'series_id': series_id,
                    'indicator_name': indicator_name,
                    'data': data['observations']
                }
    
    return {
        'success': False,
        'error': f"Failed to fetch {series_id}: Status {response.status_code}, Response: {response.text}"
    }


if __name__ == "__main__":
    INDICATORS_MAP = load_indicator_metadata()

    cnt = 0
    failed_list = []

    for series, data in INDICATORS_MAP.items():

        res = call_fred_api(series, start_date='2000-01-01', end_date=datetime.today().strftime('%Y-%m-%d'))

        # def plot_line(y, title=None):
        #     plt.plot(y.index, y['value'], color='r')
        #     if title is not None:
        #         plt.title(title)
        #     plt.xlabel('Date', fontsize=10)
        #     plt.ylabel('Value', fontsize=10)
        #     plt.legend()
        #     plt.grid(True, alpha=0.3) # alpha adjusts transparency
        #     plt.tight_layout()
        #     plt.show()

        if not res['success']:
            print(res['error'])
        else:
            cnt += 1

    print(f'{cnt} data read successfully.')