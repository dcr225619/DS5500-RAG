"""
Two evaluation functions for factuality evaluation:

1. evaluate_key_facts(answer, key_facts, api_results)
   - If the test case has "key_facts", checks whether every annotated
     data point appears in the model answer (with numeric tolerance).

2. evaluate_numeric_faithfulness(answer, api_results)
   - Regardless of key_facts, extracts ALL numbers from the answer and
     checks each one against the actual FRED data returned by the API,
     to detect hallucinated values.

One main function for these two evaluation functions:

3. evaluate_summary_factuality(answer, api_results, key_facts)
"""

import re
import json
from typing import Any

def _extract_numbers(text: str) -> list[float]:
    """
    Extract all numeric values from a text string.
    Handles formats: 26,336.3  /  3.7%  /  $1.2 trillion  /  -0.4
    Ignores: years (1900–2099), month/day numbers preceded by a month name
    Returns a deduplicated list of floats.
    """
    # strip commas inside numbers (e.g. 26,336 → 26336)
    cleaned = re.sub(r'(\d),(\d)', r'\1\2', text)

    # mark positions of year-like tokens (1900–2099) so we can skip them
    YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')
    year_spans = {m.start() for m in YEAR_RE.finditer(cleaned)}

    # month names that signal the following number is a day, not a data point
    MONTH_RE = re.compile(
        r'\b(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\b',
        re.IGNORECASE,
    )
    month_day_spans = {m.start(1) for m in MONTH_RE.finditer(cleaned)}

    seen, result = set(), []
    for m in re.finditer(r'-?\d+(?:\.\d+)?', cleaned):
        if m.start() in year_spans:
            continue
        if m.start() in month_day_spans:
            continue
        val = float(m.group())
        if val not in seen:
            seen.add(val)
            result.append(val)
    return result

def _numbers_match(a: float, b: float, rel_tol: float = 0.01, abs_tol: float = 0.1) -> bool:
    """
    True if two floats are close enough to be considered the same data point.
    Uses relative tolerance for large values, absolute for values near zero.
    """
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    
    if abs(b) < 1e-9:
        return abs(a - b) <= abs_tol
    return abs(a - b) / abs(b) <= rel_tol or abs(a - b) <= abs_tol

def _extract_floats_recursive(obj) -> list:
    """
    Recursively extract all numeric values from any nested dict/list structure.
    Skips booleans. Used to handle deeply nested analysis objects.
    """
    values = []
    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(_extract_floats_recursive(v))
    elif isinstance(obj, list):
        for v in obj:
            values.extend(_extract_floats_recursive(v))
    elif isinstance(obj, bool):
        pass  # skip booleans (True=1, False=0 pollutes the set)
    elif isinstance(obj, (int, float)):
        try:
            v = float(obj)
            if v == v:  # NaN check
                values.append(round(v, 6))
        except (ValueError, TypeError):
            pass
    elif isinstance(obj, str):
        try:
            v = float(obj)
            if v == v:
                values.append(round(v, 6))
        except (ValueError, TypeError):
            pass
    return values
 
def collect_ground_truth_values(api_results: list[dict]) -> dict[str, list[float]]:
    """
    Build a flat lookup of all numeric values returned by the FRED API,
    keyed by series_id.
 
    Handles both flat and deeply nested analysis structures by recursively
    extracting all numeric values from the analysis object and data list.
 
    Returns:
        {
            "GDP":    [26336.304, 27208.15, ...],
            "UNRATE": [3.7, 3.8, 3.6, ...],
        }
    """
    ground_truth: dict[str, list[float]] = {}
 
    for result in api_results:
        if not result.get("success"):
            continue
 
        series_id = result.get("series_id", "")
        seen = set()
        values = []
 
        def add(v, seen=seen, values=values):
            if v not in seen:
                seen.add(v)
                values.append(v)
 
        # full observations list (raw data field)
        for obs in result.get("data", []):
            try:
                v = float(obs.get("value", "nan"))
                if v == v:
                    add(round(v, 6))
            except (ValueError, TypeError):
                pass
 
        # recursively extract all numbers from analysis (handles any nesting)
        for v in _extract_floats_recursive(result.get("analysis", {})):
            add(v)
 
        if values:
            ground_truth[series_id] = values
 
    return ground_truth

# 1. Key Fact Coverage
def evaluate_key_facts(
    answer: str,
    key_facts: dict[str, dict[str, float]] | None,
    api_results: list[dict],
    rel_tol: float = 0.01,
    abs_tol: float = 0.1,
) -> dict:
    """
    Check whether annotated key facts appear in the model answer.

    Args:
        answer:     The model's final answer string.
        key_facts:  Dict from the test case, e.g.
                    {"GDP": {"2022-07-01": 26336.304}, "UNRATE": {"2024-01-01": 3.7}}
                    Pass None or {} if the test case has no key_facts.
        api_results: Raw API results from execute_tool_calls().
        rel_tol:    Relative tolerance for numeric match (default 1%).
        abs_tol:    Absolute tolerance for values near zero (default 0.1).

    Returns:
        {
            "applicable": bool,          # False if no key_facts defined
            "total_facts": int,
            "covered": int,
            "coverage_rate": float,      # covered / total_facts
            "details": [
                {
                    "series_id": "GDP",
                    "date": "2022-07-01",
                    "expected_value": 26336.304,
                    "found_in_answer": True,
                    "matched_value": 26336.304   # the number found in the answer
                },
                ...
            ]
        }
    """
    if not key_facts:
        return {
            "applicable": False,
            "total_facts": 0,
            "covered": 0,
            "coverage_rate": 1.0,
            "details": []
        }

    answer_numbers = _extract_numbers(answer)
    details = []
    covered = 0

    for series_id, date_value_map in key_facts.items():
        for date, expected_value in date_value_map.items():
            expected_value = float(expected_value)
            matched_value = None
            found = False
            for num in answer_numbers:
                num = float(num)
                if _numbers_match(num, expected_value, rel_tol, abs_tol):
                    found = True
                    matched_value = num
                    break

            if found:
                covered += 1

            details.append({
                "series_id": series_id,
                "date": date,
                "expected_value": expected_value,
                "found_in_answer": found,
                "matched_value": matched_value
            })

    total = len(details)
    return {
        "applicable": True,
        "total_facts": total,
        "covered": covered,
        "coverage_rate": round(covered / total, 4) if total > 0 else 1.0,
        "details": details
    }

# 2. Numeric Faithfulness (hallucination check)
def evaluate_numeric_faithfulness(
    answer: str,
    api_results: list[dict],
    rel_tol: float = 0.01,
    abs_tol: float = 0.1,
    ignore_below: float = 1.0,
) -> dict:
    """
    Extract all numbers from the answer and check each against the actual
    FRED data, to detect hallucinated values.

    Numbers below `ignore_below` are skipped (years, counts, percentages
    like "2.5" are evaluated; noise like "1" from "Q1" is filtered).

    Args:
        answer:       The model's final answer string.
        api_results:  Raw API results from execute_tool_calls().
        rel_tol:      Relative tolerance for match (default 1%).
        abs_tol:      Absolute tolerance for values near zero.
        ignore_below: Skip numbers smaller than this threshold (default 1.0).
                      Raise to e.g. 10.0 if you want to skip small percentages.

    Returns:
        {
            "numbers_in_answer": [26336.3, 3.7, ...],
            "total_checked": int,
            "grounded": int,             # found in API data
            "ungrounded": int,           # NOT found — potential hallucination
            "hallucination_rate": float, # ungrounded / total_checked
            "grounded_values": [26336.3, ...],
            "ungrounded_values": [99999.9, ...]   # worth inspecting manually
        }
    """
    ground_truth = collect_ground_truth_values(api_results)
    all_gt_values: list[float] = [v for vals in ground_truth.values() for v in vals]

    answer_numbers = [n for n in _extract_numbers(answer) if abs(n) >= ignore_below]

    grounded, ungrounded = [], []
    for num in answer_numbers:
        if any(_numbers_match(num, gt, rel_tol, abs_tol) for gt in all_gt_values):
            grounded.append(num)
        else:
            ungrounded.append(num)

    total = len(answer_numbers)
    return {
        "numbers_in_answer": answer_numbers,
        "total_checked": total,
        "grounded": len(grounded),
        "ungrounded": len(ungrounded),
        "hallucination_rate": round(len(ungrounded) / total, 4) if total > 0 else 0.0,
        "grounded_values": grounded,
        "ungrounded_values": ungrounded
    }

# main function
def evaluate_summary_factuality(
    answer: str,
    api_results: list[dict],
    key_facts: dict | None = None,
    rel_tol: float = 0.01,
    abs_tol: float = 0.1,
    ignore_below: float = 1.0,
    compact = True,
) -> dict:
    """
    Run both evaluations and return a combined result dict.

    Args:
        answer:       Model's final answer string.
        api_results:  Output of execute_tool_calls().
        key_facts:    From test case["key_facts"], or None.
        rel_tol:      Numeric relative tolerance.
        abs_tol:      Numeric absolute tolerance.
        ignore_below: Skip numbers below this value in faithfulness check.
        compact: Return only the coverage and hallucination rates if True. (default True).

    Returns:
        {
            "key_fact_coverage":      {...},   # from evaluate_key_facts()
            "numeric_faithfulness":   {...},   # from evaluate_numeric_faithfulness()
        }
    """
    kf = evaluate_key_facts(answer, key_facts, api_results, rel_tol, abs_tol)
    nf = evaluate_numeric_faithfulness(answer, api_results, rel_tol, abs_tol, ignore_below)

    if compact:
        return {
            "key_fact_coverage": kf['coverage_rate'],
            "numeric_faithfulness": nf['hallucination_rate'],
        }
    else:
        return {
            "key_fact_coverage": kf,
            "numeric_faithfulness": nf,
        }
