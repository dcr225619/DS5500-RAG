from fred_key import fred_key
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from metrics_computing import TimeSeriesAnalyzer
#import matplotlib.pyplot as plt

# observations
obs_endpoint = "series/observations"

# api key
api_key = fred_key
# url
base_url = "https://api.stlouisfed.org/fred/"

def load_indicator_metadata():
    with open("output_with_descriptions.json", "r", encoding="utf-8") as f:
        indicators = json.load(f)
    return {item["SERIES"]: item for item in indicators}

def call_fred_api(series_id, start_date, end_date, compact_mode=False):
    """call FRED API to get data"""
    indicators_map = load_indicator_metadata()
    indicator_name = None
    frequency = None
    units = None 
    
    if series_id in indicators_map:
        frequency = indicators_map[series_id]["PERIOD"].lower()
        indicator_name = indicators_map[series_id]["INDICATOR"]
        units = indicators_map[series_id]["UNITS"]
        description = indicators_map[series_id]['description']
    
    # make sure time range is greater than freq unit
    if frequency:
        freq_mapping = {'a': 365, 'q': 90, 'm': 30, 'w': 7, 'd': 1}
        date_delta = timedelta(days=freq_mapping[frequency])
        temp = date_delta - (datetime.strptime(start_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d"))
        if datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(end_date, "%Y-%m-%d") - date_delta:
            start_date = (datetime.strptime(start_date, "%Y-%m-%d") - temp / 2).strftime("%Y-%m-%d")
            end_date = (datetime.strptime(end_date, "%Y-%m-%d") + temp / 2).strftime("%Y-%m-%d")
            print(f'Frequency check: shift start date to {start_date}, end date to {end_date}')

    params = {
        "series_id": series_id,
        "api_key": fred_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "frequency": frequency
    }  
    
    response = requests.get(base_url + obs_endpoint, params=params)
    
    if response.status_code == 200:
        data = response.json()
        #observations = data.get("observations", [])
        try:
            analyzer = TimeSeriesAnalyzer(data["observations"])
            # analysis = analyzer.generate_summary(compact_mode=compact_mode)

            if "observations" in data and len(data["observations"]) < 2:
                # compact analysis result for single data point
                summary = analyzer.generate_summary(
                            include_full_timeseries=True,
                            recent_n_points=5,
                            include_inflections=False,
                            compact_mode=True
                        )

                return {
                    "success": True,
                    "series_id": series_id,
                    "indicator_name": indicator_name,
                    "description": description,
                    "units": units,
                    "analysis": summary
                }

            elif "observations" in data and len(data["observations"]) >= 2:
                # decide compact mode or not according to the size of data
                if compact_mode or len(data["observations"]) > 60:
                    # less than 5 data points -> compact mode
                    summary = analyzer.generate_summary(
                        include_full_timeseries=False,
                        recent_n_points=5,
                        include_inflections=False,
                        compact_mode=True
                    )
                else:
                    summary = analyzer.generate_summary(
                        include_full_timeseries=False,
                        recent_n_points=min(12, len(data["observations"])),
                        include_inflections=len(data["observations"]) >= 5,
                        compact_mode=False
                    )
                
                return {
                    "success": True,
                    "series_id": series_id,
                    "indicator_name": indicator_name,
                    "description": description,
                    "units": units,
                    "analysis": summary
                }

        except Exception as e:
            print(f"Warning: Analysis failed: {e}")
            # return original data if analysis failed
            return {
                "success": True,
                "series_id": series_id,
                "indicator_name": indicator_name,
                "description": description,
                "units": units,
                "data": data["observations"]
            }
    
    return {
        "success": False,
        "error": f"Failed to fetch {series_id}: Status {response.status_code}, Response: {response.text}"
    }


if __name__ == "__main__":
    # INDICATORS_MAP = load_indicator_metadata()

    # cnt = 0
    # failed_list = []

    # for idx, (series, data) in enumerate(INDICATORS_MAP.items()):

    #     if idx > 2:
    #         break

    #     res = call_fred_api(series, start_date="2025-01-01", end_date=datetime.today().strftime("%Y-%m-%d"))

    #     # def plot_line(y, title=None):
    #     #     plt.plot(y.index, y["value"], color="r")
    #     #     if title is not None:
    #     #         plt.title(title)
    #     #     plt.xlabel("Date", fontsize=10)
    #     #     plt.ylabel("Value", fontsize=10)
    #     #     plt.legend()
    #     #     plt.grid(True, alpha=0.3) # alpha adjusts transparency
    #     #     plt.tight_layout()
    #     #     plt.show()

    #     if not res["success"]:
    #         print(res["error"])
    #     else:
    #         cnt += 1

    # print(f"{cnt} data read successfully.")

    res = call_fred_api("WM2NS", start_date="2025-03-02", end_date='2026-03-02')
    print(res)