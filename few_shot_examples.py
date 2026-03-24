# few_shot_examples.py
FEW_SHOT_EXAMPLES = [
    {
        "question": "What's the current unemployment rate?",
        "answer": (
            "The current unemployment rate in the United States is 4.4% as of "
            "February 2026, according to the most recent data available from FRED."
        )
    },
    {
        "question": "Show me GDP data for Q1 2024",
        "answer": "The GDP for Q1 2024 is 28,708.161 billion dollars."
    },
    {
        "question": "How did unemployment and inflation change in 2024?",
        "answer": (
            "1. Unemployment Rate: The unemployment rate in 2024 showed a general "
            "increasing trend, rising from 3.7% in January to 4.2% by July, then "
            "fluctuating between 4.1% and 4.2% for the remainder of the year.\n\n"
            "2. Inflation (CPI): The Consumer Price Index started at 309.698 in "
            "January and increased steadily to 313.023 in April, remaining stable "
            "around 313 until June before gradually rising to 317.604 by December, "
            "reflecting ongoing inflationary pressures.\n\n"
            "Relationship: Both indicators showed gradual increases throughout 2024. "
            "While they appear correlated in this period, the connection is complex — "
            "unemployment often moves independently of inflation."
        )
    },
    {
        "question": "How's the real estate market in the past 2 years?",
        "answer": (
            "Here is an overview of the real estate market over the past two years:\n\n"
            "1. New Home Sales (HSN1F): Currently 745,000 units (Dec 2025), fluctuating "
            "between a low of 621,000 (Oct 2024) and a peak of 758,000 (Nov 2025), "
            "with an overall increase of 7.66%.\n\n"
            "2. Building Permits (PERMIT): Currently 1,448,000 (Dec 2025), ranging "
            "from 1,330,000 to 1,508,000, with a slight overall decrease of 1.9%.\n\n"
            "3. Mortgage Debt (MORTGAGE30US): Currently approximately $20,692,857M "
            "(Jul 2025), up 3.08% from $20,074,366M in January 2024.\n\n"
            "Summary: Sales have modestly increased, permits show slight caution in "
            "new construction, and mortgage debt continues rising — reflecting a mixed "
            "but stable outlook for the housing market."
        )
    },
    {
        "question": "What's the federal government's budget deficit trend over the last decade?",
        "answer": (
            "The federal budget deficit over the last decade shows a weak decreasing "
            "trend overall, though with significant volatility. The deficit moved from "
            "-$883.6B (Jan 2016) to -$2,052.7B (Jul 2025), with an average of "
            "approximately -$1,665B. The deficit peaked at -$4,970.5B in April 2020 "
            "during pandemic-era spending, and reached its minimum at -$742.9B in "
            "January 2022. The largest year-over-year surge (+403.8%) occurred in "
            "April 2020, while the sharpest contraction (-63.0%) came in April 2023 "
            "as emergency spending wound down."
        )
    },
]

def build_few_shot_messages() -> list[dict]:
    """Return few-shot user/assistant message pairs for summary generation."""
    messages = []
    for idx, ex in enumerate(FEW_SHOT_EXAMPLES):
        if idx in [0, 2, 3]:  # cover single/dual/multi-series three structures
            messages.append({"role": "user",      "content": ex["question"]})
            messages.append({"role": "assistant", "content": ex["answer"]})
    return messages