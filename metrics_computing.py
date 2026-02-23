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
            'percentage': round(float((end_val - start_val) / start_val * 100), 2),
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
            'recent_volatility': round(float(recent_volatility), 2),
            'earlier_volatility': round(float(earlier_volatility), 2),
            'change': round(float((recent_volatility - earlier_volatility) / earlier_volatility * 100), 2) if earlier_volatility > 0 else 0,
            'assessment': 'increased' if recent_volatility > earlier_volatility * 1.2 else 'decreased' if recent_volatility < earlier_volatility * 0.8 else 'stable'
        }
    
    def get_notable_periods(self):
        """get notable periods by identifying the greatest MoM and YoY increase and decrease"""
        values = self.df['value'].values
        dates = self.df.index
        
        notable = {}
        
        # find the greatest MoM increase and decrease
        if len(values) > 1:
            mom_changes = [(values[i] - values[i-1]) / values[i-1] * 100 
                          for i in range(1, len(values)) if values[i-1] != 0]
            if mom_changes:
                max_increase_idx = np.argmax(mom_changes) + 1
                max_decrease_idx = np.argmin(mom_changes) + 1
                
                notable['largest_mom_increase'] = {
                    'date': dates[max_increase_idx].strftime('%Y-%m-%d'),
                    'percentage': round(float(mom_changes[max_increase_idx-1]), 2),
                    'from_value': float(values[max_increase_idx-1]),
                    'to_value': float(values[max_increase_idx])
                }
                
                notable['largest_mom_decrease'] = {
                    'date': dates[max_decrease_idx].strftime('%Y-%m-%d'),
                    'percentage': round(float(mom_changes[max_decrease_idx-1]), 2),
                    'from_value': float(values[max_decrease_idx-1]),
                    'to_value': float(values[max_decrease_idx])
                }
        
        # find the greatest YoY increase and decrease
        if len(values) > 12:
            yoy_changes = [(values[i] - values[i-12]) / values[i-12] * 100 
                          for i in range(12, len(values)) if values[i-12] != 0]
            if yoy_changes:
                max_yoy_increase_idx = np.argmax(yoy_changes) + 12
                max_yoy_decrease_idx = np.argmin(yoy_changes) + 12
                
                notable['largest_yoy_increase'] = {
                    'date': dates[max_yoy_increase_idx].strftime('%Y-%m-%d'),
                    'percentage': round(float(yoy_changes[max_yoy_increase_idx-12]), 2),
                    'from_value': float(values[max_yoy_increase_idx-12]),
                    'to_value': float(values[max_yoy_increase_idx])
                }
                
                notable['largest_yoy_decrease'] = {
                    'date': dates[max_yoy_decrease_idx].strftime('%Y-%m-%d'),
                    'percentage': round(float(yoy_changes[max_yoy_decrease_idx-12]), 2),
                    'from_value': float(values[max_yoy_decrease_idx-12]),
                    'to_value': float(values[max_yoy_decrease_idx])
                }
        
        return notable
    
    def generate_integrated_timeseries(self, include_inflections=True, inflection_prominence=None):
        values = self.df['value'].values
        dates = self.df.index
        
        integrated_data = []
        
        # inflection point check
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
        
        # go through all the dates to integrate data
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
            
            # inflection point target
            data_point['inflection_type'] = inflection_dates.get(date_str, None)
            
            integrated_data.append(data_point)
        
        return integrated_data
    
    def generate_summary(self, indicator_name,
                            include_full_timeseries=False,
                            recent_n_points=12,
                            include_inflections=True,
                            inflection_prominence=None,
                            compact_mode=False):
        """
        Args:
            include_full_timeseries
            recent_n_points
            include_inflections
            inflection_prominence: significance threshold for inflection point detection
            compact_mode
        
        Returns:
            dict: summary
        """
        basic_stats = self.calculate_basic_stats()
        changes = self.calculate_changes()
        trend = self.assess_trend()
        volatility = self.detect_volatility_changes()
        notable = self.get_notable_periods()
        
        # generate complete time series data
        full_timeseries = self.generate_integrated_timeseries(
            include_inflections=include_inflections,
            inflection_prominence=inflection_prominence
        )
        
        # COMPACT MODE: only return most important information
        if compact_mode:
            summary = {
                'data_points': len(self.df),
                'time_range': {
                    'start': self.df.index[0].strftime('%Y-%m-%d'),
                    'end': self.df.index[-1].strftime('%Y-%m-%d')
                },
                'current': basic_stats['latest'],
                'extremes': {
                    'max': basic_stats['max'],
                    'min': basic_stats['min']
                },
                'trend': trend['description'],
                'total_change_pct': round(changes['total']['percentage'], 2),
                'recent_five_data': full_timeseries[-min(5, recent_n_points):]
            }
            return summary
        
        # DETAILED SUMMARY
        summary = {
            # 1. overview
            'overview': {
                'metric_name': f'{indicator_name}',
                'data_points': len(self.df),
                'time_range': {
                    'start': self.df.index[0].strftime('%Y-%m-%d'),
                    'end': self.df.index[-1].strftime('%Y-%m-%d'),
                    'span_days': (self.df.index[-1] - self.df.index[0]).days
                },
                'latest_value': {
                    'value': basic_stats['latest']['value'],
                    'date': basic_stats['latest']['date']
                }
            },
            
            # 2. key stats
            'key_statistics': {
                'current': {
                    'value': basic_stats['latest']['value'],
                    'date': basic_stats['latest']['date']
                },
                'all_time_high': {
                    'value': basic_stats['max']['value'],
                    'date': basic_stats['max']['date']
                },
                'all_time_low': {
                    'value': basic_stats['min']['value'],
                    'date': basic_stats['min']['date']
                },
                'average': round(basic_stats['mean'], 2),
                'std_deviation': round(basic_stats['std'], 2)
            },
            
            # 3. trend analysis
            'trend_analysis': {
                'overall_trend': trend['description'],
                'direction': trend['trend'],
                'strength': trend['strength'],
                'total_change': {
                    'absolute': round(changes['total']['absolute'], 2),
                    'percentage': round(changes['total']['percentage'], 2),
                    'description': f"Changed from {basic_stats['earliest']['value']} ({basic_stats['earliest']['date']}) to {basic_stats['latest']['value']} ({basic_stats['latest']['date']})"
                }
            },
            
            # 4. notable changes
            'notable_changes': notable,
            
            # 5. recent data
            'recent_data': {
                'description': f'Most recent {min(recent_n_points, len(full_timeseries))} data points',
                'data': full_timeseries[-recent_n_points:] if recent_n_points else []
            }
        }
        
        # add volatility analysis
        if volatility:
            summary['volatility_analysis'] = volatility
        
        # add inflection information
        if include_inflections:
            peaks = [d for d in full_timeseries if d['inflection_type'] == 'peak']
            troughs = [d for d in full_timeseries if d['inflection_type'] == 'trough']
            
            summary['inflection_points'] = {
                'total_count': len(peaks) + len(troughs),
                'peaks': {
                    'count': len(peaks),
                    'dates': [p['date'] for p in peaks]
                },
                'troughs': {
                    'count': len(troughs),
                    'dates': [t['date'] for t in troughs]
                },
                'most_recent_inflection': None
            }
            
            # find the most recent inflection 
            all_inflections = sorted(peaks + troughs, key=lambda x: x['date'], reverse=True)
            if all_inflections:
                summary['inflection_points']['most_recent_inflection'] = all_inflections[0]
        
        # optional: include full data
        if include_full_timeseries:
            summary['full_timeseries'] = full_timeseries
        
        return summary
    
    
    def print_summary(self):
        """
        print out summary
        """
        summary = self.generate_summary()
        print(summary)


if __name__ == '__main__':
    import json
    
    with open('./FRED data/AHEMAN.json', 'r') as f:
        data = json.load(f)
    
    analyzer = TimeSeriesAnalyzer(data=data['observations'])

    analyzer.print_summary()

    summary = analyzer.generate_summary()
    print("\nJSON Summary for LLM:")
    print(json.dumps(summary, indent=2))
