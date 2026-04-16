import unittest
import json
import pandas as pd
import numpy as np
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "../tests"))
sys.path.insert(0, os.path.join(base_dir, "../src"))

from summary_evaluation import _extract_numbers, _numbers_match, _extract_floats_recursive
from metrics_computing import TimeSeriesAnalyzer
from retrieval_accuracy_test import AccuracyEvaluator

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
        self.assertTrue(_numbers_match(27.37, 27.370000, rel_tol=0.01, abs_tol=0.1))
        self.assertTrue(_numbers_match(27.37, 27.3699999, rel_tol=0.01, abs_tol=0.1))
        self.assertTrue(_numbers_match(0.0, 0.05, rel_tol=0.01, abs_tol=0.1))
        self.assertFalse(_numbers_match(27.37, 30.0, rel_tol=0.01, abs_tol=0.1))

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
        result = _extract_floats_recursive(sample)
        self.assertEqual(result, [12.0, 4.1, 0.29, -0.69, -15.94])


# - shared data
INCREASING = [
    {"date": f"2024-0{i}-01", "value": str(i * 10)}
    for i in range(1, 7)
]
# values: 10, 20, 30, 40, 50, 60

STABLE = [
    {"date": f"2024-0{i}-01", "value": "100"}
    for i in range(1, 7)
]
# values: 100, 100, 100, 100, 100, 100

MIXED = [
    {"date": "2024-01-01", "value": "10"},
    {"date": "2024-02-01", "value": "bad"},   # non-numeric, should be dropped
    {"date": "2024-03-01", "value": "30"},
    {"date": "2024-04-01", "value": "20"},
    {"date": "2024-05-01", "value": "50"},
    {"date": "2024-06-01", "value": "40"},
]
# after dropna: 10, 30, 20, 50, 40


class TestTimeSeriesAnalyzer(unittest.TestCase):

    def test_empty_input_raises(self):
        """
        _parse_json should raise ValueError on empty input
        """
        with self.assertRaises(ValueError):
            TimeSeriesAnalyzer([])

    def test_basic_stats_known_values(self):
        """
        calculate_basic_stats should return correct max, min, mean on INCREASING
        """
        analyzer = TimeSeriesAnalyzer(INCREASING)
        stats = analyzer.calculate_basic_stats()

        self.assertEqual(stats["max"]["value"], 60.0)
        self.assertEqual(stats["max"]["date"], "2024-06-01")
        self.assertEqual(stats["min"]["value"], 10.0)
        self.assertEqual(stats["min"]["date"], "2024-01-01")
        self.assertAlmostEqual(stats["mean"], 35.0)
        self.assertEqual(stats["latest"]["value"], 60.0)
        self.assertEqual(stats["earliest"]["value"], 10.0)

    def test_assess_trend_increasing(self):
        """
        assess_trend should detect strong increasing trend on INCREASING
        """
        analyzer = TimeSeriesAnalyzer(INCREASING)
        trend = analyzer.assess_trend()

        self.assertEqual(trend["trend"], "increasing")
        self.assertEqual(trend["strength"], "strong")

    def test_assess_trend_stable(self):
        """
        assess_trend should return stable when all values are constant
        """
        analyzer = TimeSeriesAnalyzer(STABLE)
        trend = analyzer.assess_trend()

        self.assertEqual(trend["trend"], "stable")

    def test_detect_volatility_returns_none_when_too_short(self):
        """
        detect_volatility_changes should return None when len < window*2
        """
        short = INCREASING[:3]   # only 3 points, default window=6 → 3 < 6*2
        analyzer = TimeSeriesAnalyzer(short)

        self.assertIsNone(analyzer.detect_volatility_changes())

    def test_nonnumeric_values_dropped(self):
        """
        _parse_json should silently drop non-numeric rows via dropna
        """
        analyzer = TimeSeriesAnalyzer(MIXED)
        # MIXED has 6 rows, 1 non-numeric → expect 5 rows
        self.assertEqual(len(analyzer.df), 5)

    def test_generate_summary_compact_keys(self):
        """
        compact summary should contain exactly the expected top-level keys
        """
        analyzer = TimeSeriesAnalyzer(INCREASING)
        summary = analyzer.generate_summary(compact_mode=True)

        expected_keys = {"data_points", "time_range", "current", "extremes", "trend", "total_change_pct"}
        self.assertEqual(set(summary.keys()), expected_keys)
        # full_timeseries must NOT appear in compact mode
        self.assertNotIn("full_timeseries", summary)


class TestAccuracyEvaluator(unittest.TestCase):

    def setUp(self):
        # instantiate without loading any test cases or agent
        self.evaluator = AccuracyEvaluator.__new__(AccuracyEvaluator)

    def test_series_id_perfect_match(self):
        """
        F1 should be 1.0 when predicted == expected
        """
        calls = [{"series_id": "GDP"}, {"series_id": "UNRATE"}]
        result = self.evaluator.evaluate_series_id(calls, ["GDP", "UNRATE"])

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["extra"], [])

    def test_series_id_missing_and_extra(self):
        """
        F1 should be 0.5 when there are 1 missing and 1 extra series
        """
        calls = [{"series_id": "GDP"}, {"series_id": "CPIAUCSL"}]   # CPIAUCSL is extra
        result = self.evaluator.evaluate_series_id(calls, ["GDP", "UNRATE"])  # UNRATE is missing

        self.assertIn("UNRATE", result["missing"])
        self.assertIn("CPIAUCSL", result["extra"])
        self.assertEqual(result["score"], 0.5)

    def test_series_id_empty_prediction(self):
        """
        F1 should be 0.0 when no tool calls were made but series were expected
        """
        result = self.evaluator.evaluate_series_id([], ["GDP"])

        self.assertEqual(result["score"], 0.0)

    def test_date_range_no_expected(self):
        """
        evaluate_date_range should return score=1.0 when expected_date_range is None
        """
        calls = [{"series_id": "GDP", "start_date": "2024-01-01", "end_date": "2024-12-31"}]
        result = self.evaluator.evaluate_date_range(calls, None)

        self.assertEqual(result["score"], 1.0)

    def test_compare_dates_within_tolerance(self):
        """
        _compare_dates should return 1.0 when dates are within 30-day tolerance
        """
        score = self.evaluator._compare_dates("2024-01-01", "2024-01-20", tolerance_days=30)
        self.assertEqual(score, 1.0)

    def test_compare_dates_date1_after_date2(self):
        """
        _compare_dates should return 0.0 when actual end is earlier than expected end
        """
        # date1 = expected_end = 2024-12-31, date2 = actual_end = 2024-06-01
        # actual doesn't cover the expected range → score 0
        score = self.evaluator._compare_dates("2024-12-31", "2024-06-01", tolerance_days=30)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)