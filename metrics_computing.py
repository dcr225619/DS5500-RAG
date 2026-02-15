import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks, argrelextrema

"""
A tool that computes statistical metrics and trend analysis for time series data.
Reduces LLM hallucination by providing factual numerical analysis.
"""

class TimeSeriesAnalyzer:
    def __init__(self, data):
        """
        Args:
            data: list of dicts with 'date' and 'value' keys
                  Example: [{'date': '2024-01-01', 'value': '100.5'}, ...]
        """
        self.data = data.copy()
        self.df = self._parse_json()
    
    def _parse_json(self):
        """parse JSON into DataFrame"""
        if not self.data:
            raise ValueError("Input data is empty")
        
        dates = [obs['date'] for obs in self.data]
        values = [obs['value'] for obs in self.data]
        
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'value': pd.to_numeric(values, errors='coerce')
        })
        
        df = df.dropna()
        
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def calculate_basic_stats(self):
        """compute basic statistical indicators"""
        stats = {
            'max': {
                'value': float(self.df['value'].max()),
                'date': self.df['value'].idxmax().strftime('%Y-%m-%d')
            },
            'min': {
                'value': float(self.df['value'].min()),
                'date': self.df['value'].idxmin().strftime('%Y-%m-%d')
            },
            'mean': float(self.df['value'].mean()),
            'median': float(self.df['value'].median()),
            'std': float(self.df['value'].std()),
            'latest': {
                'value': float(self.df['value'].iloc[-1]),
                'date': self.df.index[-1].strftime('%Y-%m-%d')
            },
            'earliest': {
                'value': float(self.df['value'].iloc[0]),
                'date': self.df.index[0].strftime('%Y-%m-%d')
            }
        }
        return stats
    
    def calculate_changes(self):
        """compute change rate"""
        changes = {}
        
        # overall change
        start_val = self.df['value'].iloc[0]
        end_val = self.df['value'].iloc[-1]
        
        changes['total'] = {
            'absolute': float(end_val - start_val),
            'percentage': float((end_val - start_val) / start_val * 100),
            'from_date': self.df.index[0].strftime('%Y-%m-%d'),
            'to_date': self.df.index[-1].strftime('%Y-%m-%d')
        }
        
        return changes
    
    def assess_trend(self):
        """use linear regression to access overall trend"""
        x = np.arange(len(self.df))
        y = self.df['value'].values
        
        slope = np.polyfit(x, y, 1)[0]
        correlation = np.corrcoef(x, y)[0, 1]
        
        if abs(correlation) < 0.3:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        strength = "weak" if abs(correlation) < 0.5 else "moderate" if abs(correlation) < 0.8 else "strong"
        
        return {
            'trend': trend,
            'strength': strength,
            'slope': round(float(slope), 2),
            'description': f"{strength} {trend} trend"
        }
    
    def detect_volatility_changes(self, window=6):
        """detect volatility changes"""
        if len(self.df) < window * 2:
            return None
        
        rolling_std = self.df['value'].rolling(window=window).std()
        
        recent_volatility = rolling_std.iloc[-window:].mean()
        earlier_volatility = rolling_std.iloc[:window].mean()
        
        return {
            'recent_volatility': float(recent_volatility),
            'earlier_volatility': float(earlier_volatility),
            'change': float((recent_volatility - earlier_volatility) / earlier_volatility * 100) if earlier_volatility > 0 else 0,
            'assessment': 'increased' if recent_volatility > earlier_volatility * 1.2 else 'decreased' if recent_volatility < earlier_volatility * 0.8 else 'stable'
        }
    
    def generate_integrated_timeseries(self, include_inflections=True, inflection_prominence=None):
        """
        generate indicators for every date
        
        Returns:
            list of dicts: [
                {
                    'date': '2024-01-01',
                    'value': 100.5,
                    'mom_absolute': None or float,
                    'mom_percentage': None or float,
                    'yoy_absolute': None or float,
                    'yoy_percentage': None or float,
                    'inflection_type': None or 'peak' or 'trough'
                },
                ...
            ]
        """
        values = self.df['value'].values
        dates = self.df.index
        
        integrated_data = []
        
        # detect inflection
        inflection_dates = {}
        if include_inflections and len(self.df) >= 5:
            if inflection_prominence is None:
                inflection_prominence = self.df['value'].std() * 0.25
            
            peaks_idx, _ = find_peaks(values, prominence=inflection_prominence)
            troughs_idx, _ = find_peaks(-values, prominence=inflection_prominence)
            
            for idx in peaks_idx:
                inflection_dates[dates[idx].strftime('%Y-%m-%d')] = 'peak'
            for idx in troughs_idx:
                inflection_dates[dates[idx].strftime('%Y-%m-%d')] = 'trough'
        
        # integrate data
        for i in range(len(values)):
            date_str = dates[i].strftime('%Y-%m-%d')
            
            data_point = {
                'date': date_str,
                'value': float(values[i])
            }
            
            # MoM
            if i >= 1 and values[i-1] != 0:
                data_point['mom_absolute'] = round(float(values[i] - values[i-1]), 2)
                data_point['mom_percentage'] = round(float((values[i] - values[i-1]) / values[i-1] * 100), 2)
            else:
                data_point['mom_absolute'] = None
                data_point['mom_percentage'] = None
            
            # YoY
            if i >= 12 and values[i-12] != 0:
                data_point['yoy_absolute'] = round(float(values[i] - values[i-12]), 2)
                data_point['yoy_percentage'] = round(float((values[i] - values[i-12]) / values[i-12] * 100), 2)
            else:
                data_point['yoy_absolute'] = None
                data_point['yoy_percentage'] = None
            
            data_point['inflection_type'] = inflection_dates.get(date_str, None)
            
            integrated_data.append(data_point)
        
        return integrated_data
    
    def generate_summary(self, include_inflections=True, inflection_prominence=None):
        """
        Returns:
            dict with:
                - basic_stats
                - changes (overall)
                - trend
                - volatility
                - timeseries_data (inflection, mom, yoy)
        """
        summary = {
            'basic_stats': self.calculate_basic_stats(),
            'changes': self.calculate_changes(),
            'trend': self.assess_trend(),
            'data_points': len(self.df),
            'time_span': {
                'start': self.df.index[0].strftime('%Y-%m-%d'),
                'end': self.df.index[-1].strftime('%Y-%m-%d'),
                'days': (self.df.index[-1] - self.df.index[0]).days
            },
            'timeseries_data': self.generate_integrated_timeseries(
                include_inflections=include_inflections,
                inflection_prominence=inflection_prominence
            )
        }
        
        volatility = self.detect_volatility_changes()
        if volatility:
            summary['volatility'] = volatility
        
        if include_inflections:
            inflection_count = sum(1 for d in summary['timeseries_data'] if d['inflection_type'] is not None)
            peak_count = sum(1 for d in summary['timeseries_data'] if d['inflection_type'] == 'peak')
            trough_count = sum(1 for d in summary['timeseries_data'] if d['inflection_type'] == 'trough')
            
            summary['inflection_summary'] = {
                'total_count': inflection_count,
                'peak_count': peak_count,
                'trough_count': trough_count
            }
        
        return summary
    
    def print_summary(self):
        """
        print out summary
        """
        summary = self.generate_summary()
        print(summary)

# ==================== 测试代码 ====================
if __name__ == '__main__':
    # 测试用例1：从JSON文件读取
    import json
    
    # 假设你有FRED API返回的数据
    with open('./FRED data/AHEMAN.json', 'r') as f:
        data = json.load(f)
    
    analyzer = TimeSeriesAnalyzer(data=data['observations'])
    
    # 打印格式化摘要
    analyzer.print_summary()
    
    # 获取JSON格式的摘要（用于传给LLM）
    summary = analyzer.generate_summary()
    print("\nJSON Summary for LLM:")
    print(json.dumps(summary, indent=2))
