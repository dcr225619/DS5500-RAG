import json
from datetime import datetime, timedelta
from llama_api import FredLLMAgent
import pandas as pd
import time

"""
data retrieval accuracy test

scope:
1. series_id(s)
2. start_date and end_date
"""

class AccuracyEvaluator:
    def __init__(self, test_cases_file=None):
        """
        Args:
            test_cases_file: json file path2
        """
        self.agent = FredLLMAgent(verbose=False)
        self.test_cases = []
        
        if test_cases_file:
            self.load_test_cases(test_cases_file)
    
    def load_test_cases(self, filepath):
        """load test cases from json file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.test_cases = json.load(f)
    
    def add_test_case(self, question, expected_series_ids, 
                     expected_date_range=None, description=""):
        """
        add test case
        
        Args:
            question
            expected_series_ids: e.g. ["GDP", "UNRATE"]
            expected_date_range:
                e.g. {"start": "2020-01-01", "end": "2024-12-31"}
                or {"relative_start": "1y", "relative_end": "today"}
            description
        """
        self.test_cases.append({
            'question': question,
            'expected_series_ids': expected_series_ids,
            'expected_date_range': expected_date_range,
            'description': description
        })
    
    def evaluate_series_id(self, actual_calls, expected_series_ids):
        """
        series_id accuracy
        
        Returns:
            dict: {
                'score': float (0-1),
                'correct_count': int,
                'missing': list of str,
                'extra': list of str
            }
        """
        actual_ids = [call['series_id'] for call in actual_calls]
        expected_set = set(expected_series_ids)
        actual_set = set(actual_ids)
        
        correct = expected_set & actual_set
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        
        # calculate F1 score for data retrieval
        if not expected_set:
            score = 1.0 if not actual_set else 0.0
        else:
            precision = len(correct) / len(actual_set) if actual_set else 0
            recall = len(correct) / len(expected_set)
            # F1 score
            score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'score': round(score, 2),
            'correct_count': len(correct),
            'correct': list(correct),
            'missing': list(missing),
            'extra': list(extra)
        }
    
    def evaluate_date_range(self, actual_calls, expected_date_range):
        """
        date ranges accuracy
        
        Returns:
            dict: {
                'score': float (0-1),
                'details': str
            }
        """
        if not expected_date_range or not actual_calls:
            return {'score': 1.0, 'details': 'No date range to evaluate'}
        
        scores = []
        details = []
        
        for call in actual_calls:
            actual_start = call.get('start_date', '')
            actual_end = call.get('end_date', '')
            
            # deal with relative date
            if 'relative_start' in expected_date_range:
                relative = expected_date_range['relative_start']
                if relative.endswith('y'):
                    years = int(relative[:-1])
                    expected_start = (datetime.today() - timedelta(days=365*years)).strftime('%Y-%m-%d')
                elif relative.endswith('m'):
                    months = int(relative[:-1])
                    expected_start = (datetime.today() - timedelta(days=30*months)).strftime('%Y-%m-%d')
                else:
                    expected_start = expected_date_range.get('start', '')
            else:
                expected_start = expected_date_range.get('start', '')
            
            if expected_date_range.get('relative_end') == 'today':
                expected_end = datetime.today().strftime('%Y-%m-%d')
            else:
                expected_end = expected_date_range.get('end', '')
            
            # data date
            start_match = self._compare_dates(actual_start, expected_start, tolerance_days=30)
            end_match = self._compare_dates(actual_end, expected_end, tolerance_days=30)
            
            call_score = (start_match + end_match) / 2
            scores.append(call_score)
            
            details.append(f"series_id={call['series_id']}: "
                          f"start: (api call: {actual_start} vs expected: {expected_start}), "
                          f"end: (api call: {actual_end} vs expected: {expected_end})")
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            'score': round(avg_score, 2),
            'details': '; '.join(details)
        }
    
    def _compare_dates(self, actual, expected, tolerance_days=30):
        """
        compare two dates and return the relavant score
        
        Returns:
            float: 0-1, 1 represents complete match
        """
        if not actual or not expected:
            return 0.0
        
        try:
            actual_date = datetime.strptime(actual, '%Y-%m-%d')
            expected_date = datetime.strptime(expected, '%Y-%m-%d')
            
            diff_days = abs((actual_date - expected_date).days)
            
            if diff_days == 0:
                return 1.0
            elif diff_days <= tolerance_days:
                return 1.0 - (diff_days / tolerance_days)  # score deducted if within tolerance
            else:
                return 0.0
        except:
            return 0.0
    
    def evaluate_single_case(self, test_case):
        """
        evaluate a single test case
        extraction:
            dict: {
                    'success': bool,
                    'tool_calls': list of dict with {series_id, start_date, end_date},
                    'raw_response': dict,
                    'error': str (if failed)
                }
        Returns:
            {
                'question_id',
                'success': boolean,
                'actual_calls': actual_calls,
                'series_id_evaluation': series_eval,
                'date_range_evaluation': date_eval,
                'overall_score': round(overall_score, 2)
            }
        """
        question = test_case['question']
        question_id = test_case['question_id']
        expected_series_ids = test_case.get('expected_series_ids', [])
        expected_date_range = test_case.get('expected_date_range', {})
        
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"Expected series_ids: {expected_series_ids}")
        
        extraction = self.agent.extract_tool_calls(question)
        
        if not test_case.get("tool_call_required", True):
            # non-tool-call-required question:
            # if no tool call extracted -> score=1
            # if tool call extracted -> score=0
            made_tool_call = extraction['success'] and bool(extraction.get('tool_calls'))
            if made_tool_call:
                return {
                    'question_id': question_id,
                    'success': False,
                    'error': 'Model incorrectly made a tool call for a non-tool-call question',
                    'series_id_score': 0,
                    'date_range_score': 0,
                    'overall_score': 0
                }
            else:
                return {
                    'question_id': question_id,
                    'success': True,
                    'series_id_evaluation': {'score': 1.0, 'correct': [], 'missing': [], 'extra': []},
                    'date_range_evaluation': {'score': 1.0, 'details': 'No tool call expected or made'},
                    'overall_score': 1.0
                }

        # tool-call-required question but no tool call extracted
        if not extraction['success']:
            return {
                'question_id': question_id,
                'success': False,
                'error': extraction.get('error', 'Tool call extraction failed'),
                'series_id_score': 0,
                'date_range_score': 0,
                'overall_score': 0
            }

        actual_calls = extraction['tool_calls']
        
        series_eval = self.evaluate_series_id(actual_calls, expected_series_ids)
        date_eval = self.evaluate_date_range(actual_calls, expected_date_range)
        overall_score = series_eval['score'] * 0.5 + date_eval['score'] * 0.5
        
        return {
            'question_id': question_id,
            'success': True,
            'actual_calls': actual_calls,
            'series_id_evaluation': series_eval,
            'date_range_evaluation': date_eval,
            'overall_score': round(overall_score, 2)
        }
    
    def run_all_tests(self):
        """
        run all tests
        """
        results = []
        
        print(f"\n{'#'*60}")
        print(f"Running {len(self.test_cases)} test cases...")
        print(f"{'#'*60}")
        
        for i, test_case in enumerate(self.test_cases, 1):
            if i >= 327:
                break
            print(f"\n[Test {i}/{len(self.test_cases)}]")
            result = self.evaluate_single_case(test_case)
            results.append(result)
        
        successful_tests = [r for r in results if r['success']]
        
        if not successful_tests:
            print("\nAll tests failed!")
            return {'results': results, 'summary': None}

        
        avg_series_score = sum(
            r['series_id_evaluation']['score']
            for r in successful_tests
            if 'series_id_evaluation' in r
        ) / len(successful_tests)

        avg_date_score = sum(
            r['date_range_evaluation']['score']
            for r in successful_tests
            if 'date_range_evaluation' in r
        ) / len(successful_tests)

        avg_overall_score = sum(r['overall_score'] for r in successful_tests) / len(successful_tests)

        
        tool_call_tests = [r for r in results if self.test_cases[results.index(r)].get('tool_call_required', True)]
        non_tool_tests  = [r for r in results if not self.test_cases[results.index(r)].get('tool_call_required', True)]

        summary = {
            'total_tests': len(self.test_cases),
            'successful_tests': len(successful_tests),
            'failed_tests': len(self.test_cases) - len(successful_tests),
            'tool_call_tests': len(tool_call_tests),
            'tool_call_passed': sum(1 for r in tool_call_tests if r['success']),
            'non_tool_call_tests': len(non_tool_tests),
            'non_tool_call_passed': sum(1 for r in non_tool_tests if r['success']),
            'avg_series_id_score': round(avg_series_score, 2),
            'avg_date_range_score': round(avg_date_score, 2),
            'avg_overall_score': round(avg_overall_score, 2)
        }
        
        print(f"\n{'#'*60}")
        print("SUMMARY")
        print(f"{'#'*60}")
        print(f"Total tests:        {summary['total_tests']}")
        print(f"Successful:         {summary['successful_tests']}")
        print(f"Failed:             {summary['failed_tests']}")
        print(f"\nBreakdown:")
        print(f"  Tool-call tests:      {summary['tool_call_tests']} "
              f"(passed: {summary['tool_call_passed']})")
        print(f"  Non-tool-call tests:  {summary['non_tool_call_tests']} "
              f"(passed: {summary['non_tool_call_passed']})")
        print(f"\nAverage Scores (successful tests only):")
        print(f"  Series ID:   {summary['avg_series_id_score']}")
        print(f"  Date Range:  {summary['avg_date_range_score']}")
        print(f"  Overall:     {summary['avg_overall_score']}")
        print(f"{'#'*60}\n")
        
        return {
            'results': results,
            'summary': summary
        }
    
    def export_results(self, results, filepath='files/evaluation_results.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults exported to {filepath}")



if __name__ == "__main__":
    
    start = time.time()

    evaluator = AccuracyEvaluator()
    
    # add test cases
    path = 'data/QA2.json'
    evaluator.load_test_cases(path)
    
    results = evaluator.run_all_tests()
    
    evaluator.export_results(results,'files/origin/evaluation_results_compact6.json' )

    print(f'Execution time: {time.time() - start: .2f}')