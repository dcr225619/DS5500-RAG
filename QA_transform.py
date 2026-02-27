import json
from datetime import datetime, timedelta
import time
from fred_api import load_indicator_metadata

"""
this script is used for transforming tool calls QA's in QA.json file 
into llm conversations that are suitable for llama 3.1 fine-tuning.
"""

INDICATORS_MAP = load_indicator_metadata()

with open('files/indicator_guide_compact.txt', encoding='utf-8') as f:
    indicator_mapping = f.read()

today = datetime.today()

INDICATOR_GUIDE = indicator_mapping + f"""
Note: (M)=Monthly, (Q)=Quarterly, (W)=Weekly, (D)=Daily, (Y)=Yearly

When asked about economic data, use the get_fred_data function with the appropriate series_id.

You will receive data from ALL tool calls.

IMPORTANT: 
1. Always specify start_date and end_date based on the user's question. Always use YYYY-MM-DD format, NOT relative dates like "-2y"
2. If no time period specified, use recent 1 year by default
3. Each series_id should only be called ONCE with a single continuous date range.
   - WRONG: calling GDP twice with 2018-2020 and 2021-2023
   - RIGHT: calling GDP once with 2018-2023
4. Today is {today.strftime('%Y-%m-%d')}
"""

with open('data/QA.json', encoding='utf-8') as f:
    file = json.load(f)

results = []
for idx, q in enumerate(file):
    question = q['question']

    # edge case: don't need tool call
    # choose direct responds according to respond type
    if not q.get("tool_call_required", True):
        response_type = q.get("response_type", "direct_answer")

        if response_type == "direct_answer":
            assistant_content = q.get("description", "I can answer that directly without looking up any data.")

        elif response_type == "indicator_recommendation":
            assistant_content = q.get("description", "I can suggest some relevant indicators for that topic.")

        elif response_type == "out_of_scope":
            assistant_content = (
                "That data isn't available in my current FRED indicator set. "
                "I can only retrieve macroeconomic series such as GDP, CPI, unemployment, "
                "interest rates, and trade data. Please try a different question."
            )

        elif response_type == "clarification_needed":
            assistant_content = (
                "Could you clarify your request? I need a bit more detail — "
                "for example, which indicator or country you're interested in, "
                "and what time period you'd like to look at."
            )

        elif response_type == "out_of_time_range":
            assistant_content = (
                "I'm unable to retrieve that data — it either falls in the future "
                "or predates the available FRED data coverage. "
                "Please try a different time period."
            )

        else:
            assistant_content = q.get("description", "I'm unable to process that request.")

        template = {
            "conversations": [
                {
                    "role": "system",
                    "content": f"You are an economic data assistant with access to FRED API. {INDICATOR_GUIDE}"
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]
        }
        results.append(template)
        continue

    series_ids = q.get("expected_series_ids", [])
    dates = q.get("expected_date_range", {})

    # edge case: no tool call (wrong response for tool-call-required question)
    if not series_ids:
        alternatives = q.get("suggested_alternatives", [])

        if alternatives:
            alt_descriptions = []
            for sid in alternatives:
                if sid in INDICATORS_MAP:
                    name = INDICATORS_MAP[sid]['INDICATOR']
                    alt_descriptions.append(f"{name} ({sid})")
            alt_text = ", ".join(alt_descriptions) if alt_descriptions else "no close alternatives available"
            assistant_content = (
                f"I don't have a direct indicator for that in my available data. "
                f"The closest alternatives would be {alt_text}. "
                f"Would you like me to retrieve any of these instead?"
            )
        else:
            assistant_content = (
                "I don't have a direct indicator for that in my available data. "
                "Please try another question."
            )

        template = {
            "conversations": [
                {
                    "role": "system",
                    "content": f"You are an economic data assistant with access to FRED API. {INDICATOR_GUIDE}"
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]
        }
        results.append(template)
        continue 

    # deal with relative dates
    if 'relative_start' in dates:
        relative = dates['relative_start']
        if relative.endswith('y'):
            years = int(relative[:-1])
            expected_start = (today - timedelta(days=365 * years)).strftime('%Y-%m-%d')
        elif relative.endswith('m'):
            months = int(relative[:-1])
            expected_start = (today - timedelta(days=30 * months)).strftime('%Y-%m-%d')
        else:
            expected_start = dates.get('start', '')
    else:
        expected_start = dates.get('start', '')

    if dates.get('relative_end') == 'today':
        expected_end = today.strftime('%Y-%m-%d')
    else:
        expected_end = dates.get('end', '')

    # combine all the tool calls
    tool_calls_content = "\n".join([
        f'<tool_call>\n{json.dumps({"name": "get_fred_data", "arguments": {"series_id": sid, "start_date": expected_start, "end_date": expected_end}})}\n</tool_call>'
        for sid in series_ids
    ])

    template = {
        "conversations": [
            {
                "role": "system",
                "content": f"You are an economic data assistant with access to FRED API. {INDICATOR_GUIDE}"
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": tool_calls_content
            }
        ]
    }

    results.append(template)

# file
with open('data/QA_finetune2.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)