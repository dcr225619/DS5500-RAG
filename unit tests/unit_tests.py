import unittest
import json
import pandas as pd
import numpy as np
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
# add the current directory to Python's path so that imports like `from fred_key import fred_key` can locate modules in the same directory
sys.path.insert(0, os.path.join(base_dir, "../tests"))

from summary_evaluation import _extract_numbers, _numbers_match, _extract_floats_recursive, 

class TestExtractNumbers(unittest.TestCase):
    def test_extract_numbers(self):
        """
        Test `_extract_numbers()` which extracts numeric claims from LLM's response.
        """
        sample = "Over the past 5 years, manufacturing wages, as indicated by average hourly earnings and average weekly hours in the manufacturing sector, have experienced notable changes:\n\n### Average Hourly Earnings (AHEMAN)\n- **March 2021**: $23.35\n- **February 2026**: $29.74\n- **Total Change**: An increase of $6.39 or approximately **27.37%**.\n- **Trend**: There has been a strong increasing trend in average hourly earnings over this period.\n- **Notable Changes**:\n  - This time series reached an all-time high of **$29.74** in February 2026.\n  - The average hourly earnings have shown consistent growth, reflecting wage increases across the manufacturing sector.\n\n### Average Weekly Hours (AWHMAN)\n- **March 2021**: 41.6 hours\n- **February 2026**: 41.5 hours\n- **Total Change**: A slight decrease of **0.1 hours**, representing a **0.24%** decline.\n- **Trend**: The average weekly hours have shown a weak decreasing trend.\n- **Notable Changes**:\n  - The highest recorded average weekly hours were **41.6** in March 2021, while the lowest was **40.3** in December 2023.\n  - Despite fluctuations, the average weekly hours have remained relatively stable around 41.5 hours, indicating sustained working hours despite wage increases.\n\n### Summary\nOverall, manufacturing wages have increased significantly, with hourly earnings rising sharply, while average weekly hours have seen a minor decline. This suggests that even though workers are earning more per hour, the amount of time spent working each week has slightly decreased, which could reflect changes in labor demand or efficiency improvements in the manufacturing sector."
        result = _extract_numbers(sample)
        self.assertEqual(result, [5.0, 23.35, 29.74, 6.39, 27.37, 41.6, 41.5, 0.1, 0.24, 40.3])

    def test_numbers_match(self):
        """
        Test `_numbers_match()` which compares numeric claims from LLM's response with raw api outputs.
        """
        sample1, sample2 = 27.37, 27.370000
        self.assertEqual(sample1, sample2)
        sample3, sample4 = 27.37, 27.3699999
        self.assertEqual(sample3, sample4)
        sample5, sample6 = 0.0000001, 0.00001
        self.assertEqual(sample5, sample6)

    def test_extract_floats_recursive(self):
        """
        Test `_extract_floats_recursive()` which extracts numeric claims from fred_api outputs.
        """
        sample = [
            {
                "success": True,
                "series_id": "FEDFUNDS",
                "analysis": {
                    "overview": {"data_points": 12},
                    "key_statistics": {"average": 4.1, "std_deviation": 0.29},
                    "trend_analysis": {"total_change": {"absolute": -0.69, "percentage": -15.94}},
                }
            }
        ]
        result = _extract_floats_recursive(json.loads(sample))
        self.assertEqual(result, [12.0, 4.1, 0.29, -0.69, -15.94])


if __name__ == "__main__":
    unittest.main()