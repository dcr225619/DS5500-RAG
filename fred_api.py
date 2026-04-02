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
        "series_id":         series_id,
        "api_key":           fred_key,
        "file_type":         "json",
        "observation_start": start_date,
        "observation_end":   end_date,
    }
    if frequency:
        params["frequency"] = frequency
    
    response = requests.get(base_url + obs_endpoint, params=params)
    
    if response.status_code == 200:
        data = response.json()
        observations = data.get("observations", [])

        # filter out value == "."
        observations = [o for o in observations if o.get("value", ".") != "."]

        if not observations:
            return {
                "success": False,
                "series_id": series_id,
                "error": f"No data returned for {series_id} in range {start_date} to {end_date}"
            }

        try:
            analyzer = TimeSeriesAnalyzer(observations)

            if len(observations) < 5:
                summary = analyzer.generate_summary(
                    include_full_timeseries=False,
                    include_inflections=False,
                    compact_mode=True
                )
            elif compact_mode or len(observations) < 180:
                summary = analyzer.generate_summary(
                    include_full_timeseries=False,
                    include_inflections=True,
                    compact_mode=True,
                )
            else:
                summary = analyzer.generate_summary(
                    include_full_timeseries=False,
                    include_inflections=True,
                    compact_mode=False
                )

            return {
                "success": True,
                "series_id": series_id,
                "indicator_name": indicator_name,
                "description": description,
                "units": units,
                "analysis": summary,
                "raw_observations": observations,   # always keep the original data available for use in charts
            }

        except Exception as e:
            print(f"Warning: Analysis failed: {e}")
            return {
                "success": True,
                "series_id": series_id,
                "indicator_name": indicator_name,
                "description": description,
                "units": units,
                "data": observations,
                "raw_observations": observations,   # ← 同上
            }

    return {
        "success": False,
        "series_id": series_id,
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