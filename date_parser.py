import dateparser
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen":17, "eighteen": 18, 
    "nineteen": 19, "twenty": 20,
}

def normalize_number_words(text: str) -> str:
    """Replace English number words with digits for easier regex matching."""
    for word, num in WORD_TO_NUM.items():
        text = re.sub(rf"\b{word}\b", str(num), text)
    return text

def parse_date_range(question: str):
    """
    Extract start/end date from a natural language question.
    Returns (start_date, end_date) as YYYY-MM-DD strings, or (None, None) if uncertain.
 
    Design principle: only return a date range when the question contains an
    explicit, unambiguous time expression. Any vague/implicit signal is left
    for the LLM to handle.
    """
    today = datetime.today()
    text = normalize_number_words(question.lower())
 
    # ------------------------------------------------------------------
    # 1. Half-year: "first half of 2022", "second half of 2020"
    # ------------------------------------------------------------------
    half_map = {
        r"first half":  (1, 6),
        r"second half": (7, 12),
        r"first half of the year": (1, 6),
        r"second half of the year": (7, 12),
    }
    for label, (start_month, end_month) in half_map.items():
        pattern = rf"{label}(?:\s+of)?\s+(\d{{4}})"
        m = re.search(pattern, text)
        if m:
            year = int(m.group(1))
            start = datetime(year, start_month, 1)
            end = datetime(year, end_month, 1) + relativedelta(months=1) - timedelta(days=1)
            end = min(end, today)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
 
    # ------------------------------------------------------------------
    # 2. Quarter patterns: "Q1 2024", "Q3 and Q4 2022", "first quarter of 2023"
    # ------------------------------------------------------------------
    q_map = {
        "q1": 1,  "q2": 4,  "q3": 7,  "q4": 10,
        "first quarter":  1, "second quarter": 4,
        "third quarter":  7, "fourth quarter": 10,
    }
 
    all_years = [int(y) for y in re.findall(r"\b(\d{4})\b", text)]
    
    # use [] to store all quarter indicators
    quarters_found = []
    for label, month in q_map.items():
        # label immediately followed by a year
        pattern_with_year = rf"{re.escape(label)}[\s,of]*(\d{{4}})"
        for m in re.finditer(pattern_with_year, text):
            year = int(m.group(1))
            q_start = datetime(year, month, 1)
            q_end = q_start + relativedelta(months=3) - timedelta(days=1)
            quarters_found.append((q_start, q_end))
 
        # label without an attached year — use the sole year in question
        if not quarters_found or not any(
            re.search(pattern_with_year, text) for _, _ in [(label, month)]
        ):
            pattern_no_year = rf"{re.escape(label)}(?![\s,of]*\d{{4}})"
            if re.search(pattern_no_year, text) and len(all_years) == 1:
                year = all_years[0]
                q_start = datetime(year, month, 1)
                q_end = q_start + relativedelta(months=3) - timedelta(days=1)
                quarters_found.append((q_start, q_end))
 
    if quarters_found:
        start = min(q[0] for q in quarters_found)
        end = min(max(q[1] for q in quarters_found), today)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
 
    # ------------------------------------------------------------------
    # 3. "since YEAR" / "since the start of YEAR" → (YEAR-01-01, today)
    #    "from YEAR to YEAR" → full range
    # ------------------------------------------------------------------
    # "since [the start/beginning of] YEAR"
    m = re.search(r"since(?:\s+the\s+(?:start|beginning)\s+of)?\s+(\d{4})", text)
    if m:
        year = int(m.group(1))
        start = datetime(year, 1, 1)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
 
    # "from YEAR to YEAR" or "from YEAR through YEAR"
    m = re.search(r"from\s+(\d{4})\s+(?:to|through)\s+(\d{4})", text)
    if m:
        start = datetime(int(m.group(1)), 1, 1)
        end = datetime(int(m.group(2)), 12, 31)
        end = min(end, today)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
 
    # ------------------------------------------------------------------
    # 4. Explicit relative ranges with a number:
    #    "past 2 years", "last 6 months", "over the past 18 months",
    #    "past decade" (decade = 10 years)
    # ------------------------------------------------------------------
    m = re.search(
        r"(?:past|last|over the past|over the last)\s+"
        r"(\d+)\s+(year|month|week)s?",
        text
    )
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "year":
            start = today - relativedelta(years=n)
        elif unit == "month":
            start = today - relativedelta(months=n)
        else:
            start = today - timedelta(weeks=n)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
 
    # "past decade" / "last decade"
    if re.search(r"(?:past|last|over the past)\s+decade", text):
        start = today - relativedelta(years=10)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
 
    # ------------------------------------------------------------------
    # 5. Explicit relative ranges WITHOUT a number:
    #    "current year", "this year", "current", "recently", "now", "lately", "last year", "past year"
    # ------------------------------------------------------------------
    if re.search(r"\b(?:this year|current year|current|recently|now|lately)\b", text):
        start = datetime(today.year, 1, 1)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
 
    if re.search(r"\b(?:last year|past year)\b", text):
        start = today - relativedelta(years=1)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
 
    # ------------------------------------------------------------------
    # 6. Full single year: "in 2024", "during 2019", "for all of 2022",
    #    "what was X in 2023", etc.
    #    Only triggers when there is exactly ONE year token — two or more
    #    years without an explicit range connector are ambiguous, so we
    #    leave them for the LLM.
    # ------------------------------------------------------------------
    if len(all_years) == 1:
        year = all_years[0]
        start = datetime(year, 1, 1)
        end = min(datetime(year, 12, 31), today)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
 
    # ------------------------------------------------------------------
    # 7. Anything else (vague words like "recently", "now", "lately",
    #    "right now", two bare years without a connector, etc.)
    #    → return (None, None) and let the LLM decide.
    # ------------------------------------------------------------------
    return None, None

if __name__ == '__main__':
    import json

    with open("data/QA_test.json", encoding="utf-8") as f:
        file = json.load(f)

    for idx, question in enumerate(file):
        question_id = question["question_id"]
        print(f"Question {question_id}: {question["question"]}")
        result = parse_date_range(question['question'])
        expected = question["expected_date_range"]
        status = ""
 
        # Simple pass/fail check
        if expected is None and result == (None, None):
            status = "✓"
        elif expected is not None and result != (None, None):
            status = "✓"
        else:
            status = "✗"
 
        print(f"[{status}] {question_id}: {question['question']}")
        if status == "✗":
            print(f"     expected : {expected}")
            print(f"     got      : {result}")
        
        