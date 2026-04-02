# applying semantic retriever for llama_api.py

import requests
from fred_key import fred_key
from fred_api import load_indicator_metadata, call_fred_api
from series_retriever import SeriesRetriever
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
from few_shot_examples import build_few_shot_messages
import re

OLLAMA_URL = "http://localhost:11434/api/chat"

retriever = SeriesRetriever()

GUIDE_SUFFIX = f"""
Note: (M)=Monthly, (Q)=Quarterly, (W)=Weekly, (D)=Daily, (Y)=Yearly

When asked about economic data, use the get_fred_data function with the appropriate series_id. Otherwise, directly answer the question.

For vague or unspecified request, ask for further clarification and explanation.

You will receive data from ALL tool calls.

IMPORTANT:
1. Always specify start_date and end_date based on the user's question. Always use YYYY-MM-DD format, NOT relative dates like "-2y"
2. If no time period specified, use recent 1 year by default
3. Each series_id should only be called ONCE with a single continuous date range.
   - WRONG: calling GDP twice with 2018-2020 and 2021-2023
   - RIGHT: calling GDP once with 2018-2023
4. Today is {datetime.today().strftime('%Y-%m-%d')}
"""

def build_indicator_guide(question: str, top_k: int = 8) -> str:
    """
    Dynamically build a prompt section with only the most relevant series
    for the given question, instead of listing all 90+ series every time.
    """
    return retriever.build_prompt_section(question, top_k=top_k) + GUIDE_SUFFIX

TOOLS = [
    {
        "name": "get_fred_data",
        "description": "Gain economic data from FRED API, for example GDP, unemployment rate, CPI etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "series_id": {
                    "type": "string",
                    "description": "FRED data series ID, for example: GDP(US Gross Domestic Product), UNRATE(Civilian Total Unemployment Rate), CPIAUCSL(Consumer Price Index: Seasonally Adj.)"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format. For 'current/recent' queries, use 1 years ago. For specific periods, use the exact start date."
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format. Usually today's date unless a specific period is requested."
                }
            },
            "required": ["series_id", "start_date", "end_date"]
        }
    }
]

def fix_date_parameters(start_date, end_date):
    today = datetime.today()

    # 1. format check
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except:
        print(f"  Invalid date format, defaulting to past 1 year")
        return (today - timedelta(days=365)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    # 2. end_date cannot be later than today
    if end_dt > today:
        print(f"  End date {end_date} is in the future, capping to today")
        end_dt = today

    # 3. start == end -> start adjusted to 1 year ago
    if start_dt.date() == end_dt.date():
        print(f"  Detected same-day query, expanding to 1 year")
        start_dt = end_dt - timedelta(days=365)

    # 4. start > end -> start adjusted to 1 year ago
    elif start_dt > end_dt:
        print(f"  Start date later than end date, adjusting to 1 year ago")
        start_dt = end_dt - timedelta(days=365)

    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

def call_fred_api_with_fallback(series_id, start_date, end_date, max_retries=1, compact_mode=False):
    """
    call FRED API, fall back if fail
    
    strategy:
    1. try with original dates
    2. if no data returns, push start_date 1 year earlier 
    
    Args:
        series_id: FRED series ID
        start_date
        end_date
        max_retries
    
    Returns:
        dict: API results
    """
    
    for attempt in range(max_retries):
        result = call_fred_api(series_id, start_date, end_date, compact_mode=compact_mode)
        
        if result["success"] and result.get("data"):
            if attempt > 0:
                print(f"  Retry {attempt} succeeded with date range: {start_date} to {end_date}")
            return result
        
        # fall back
        if attempt < max_retries - 1:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                start_dt = start_dt - timedelta(days=365)
                start_date = start_dt.strftime("%Y-%m-%d")
                print(f"  No data found, retrying with start_date: {start_date}")
            except:
                break
    
    return result

# relative date resolver e.g.
# "-1y", "-2y", "-18m", "-6m", "-3y", "today", "now", "-1y6m"
_REL_PATTERN = re.compile(
    r"""^
    (?P<sign>[-+])?          # optional leading sign
    (?:(?P<years>\d+)\s*y)?  # e.g. 1y  / 2y
    (?:(?P<months>\d+)\s*m)? # e.g. 6m  / 18m
    (?:(?P<days>\d+)\s*d)?   # e.g. 30d
    $""",
    re.IGNORECASE | re.VERBOSE,
)

_TODAY_ALIASES = {"today", "now", "current", "present"}

def resolve_relative_date(value: str, reference: datetime | None = None) -> str | None:
    """
    Try to interpret *value* as a relative date offset and return an absolute
    YYYY-MM-DD string.  Return None if *value* does not look relative.

    Supported formats (case-insensitive, leading/trailing whitespace stripped):
        "today" / "now" / "current" / "present"  → today
        "-1y"   / "-2y"  / "-5y"                 → N years ago
        "-6m"   / "-18m"                          → N months ago
        "-30d"                                    → N days ago
        "-1y6m"                                   → 1 year + 6 months ago
        "+1y"                                     → 1 year in the future (capped to today)

    Returns None for strings that are already absolute YYYY-MM-DD dates,
    or for any unrecognised format.
    """
    if not isinstance(value, str):
        return None

    clean = value.strip().lower()

    # 1. today aliases
    if clean in _TODAY_ALIASES:
        ref = reference or datetime.today()
        return ref.strftime("%Y-%m-%d")

    # 2. not relative expressions
    try:
        datetime.strptime(clean, "%Y-%m-%d")
        return None
    except ValueError:
        pass

    # 3. regex match for offset patterns in relative expressions
    # detect and strip leading sign, default direction is negative (past)
    if clean.startswith("+"):
        sign, body = 1, clean[1:]
    elif clean.startswith("-"):
        sign, body = -1, clean[1:]
    else:
        sign, body = -1, clean   # bare "1y" / "6m" also treated as "past"

    m = _REL_PATTERN.match(body)
    if not m or not any(m.group(k) for k in ("years", "months", "days")):
        return None

    ref = reference or datetime.today()
    delta = relativedelta(
        years=sign  * int(m.group("years")  or 0),
        months=sign * int(m.group("months") or 0),
        days=sign   * int(m.group("days")   or 0),
    )
    result_dt = ref + delta

    # cap future dates to today
    today = reference or datetime.today()
    if result_dt > today:
        result_dt = today

    return result_dt.strftime("%Y-%m-%d")


class FredLLMAgent:
    def __init__(self, model="llama3.2", api_url=OLLAMA_URL, verbose=True, top_k=5, few_shot=False):
        self.model = model
        self.api_url = api_url
        self.verbose = verbose  # print process or not
        self.top_k = top_k
        self.few_shot = few_shot  # use few-shot prompting for summary generation or not
        
    def call_llm(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS,
            "stream": False
        }
        
        response = requests.post(self.api_url, json=payload, timeout=600)
        return response.json()
    
    def _resolve_tool_call_dates(self, args: dict) -> tuple[str, str]:
        """
        Determine the final (start_date, end_date) for a single tool call.

        Priority order
        ──────────────
        1. Absolute YYYY-MM-DD dates from the LLM  → used directly
        2. Relative dates from the LLM (e.g. "-1y", "-18m", "today")
           → resolved to absolute via resolve_relative_date()
        3. fix_date_parameters() sanitisation as a final safety net

        Args:
            args: the raw arguments dict from the LLM tool call

        Returns:
            (start_date, end_date) both guaranteed to be valid YYYY-MM-DD strings
        """
        raw_start = (args.get("start_date") or args.get("start", "")).strip()
        raw_end   = (args.get("end_date")   or args.get("end",   "")).strip()

        # resolve_relative_date() returns None when the string is already
        # an absolute date, so we only substitute when it actually found
        # a relative pattern.
        resolved_start = resolve_relative_date(raw_start)
        resolved_end   = resolve_relative_date(raw_end)

        start_date = resolved_start if resolved_start is not None else raw_start
        end_date   = resolved_end   if resolved_end   is not None else raw_end

        if self.verbose:
            if resolved_start is not None:
                print(f"  [DateResolver] Relative start '{raw_start}' → '{start_date}'")
            if resolved_end is not None:
                print(f"  [DateResolver] Relative end   '{raw_end}'   → '{end_date}'")

        # Final sanitisation: catches empty strings, bad formats, future
        # end dates, start > end, etc.
        start_date, end_date = fix_date_parameters(start_date, end_date)
        return start_date, end_date

    def extract_tool_calls(self, question):
        """
        extract tool calls without execution
        
        Returns:
            dict: {
                "success": bool,
                "tool_calls": list of dict with {series_id, start_date, end_date},
                "raw_response": dict,
                "error": str (if failed)
            }
        """
        # dynamically search for top k series
        guide = build_indicator_guide(question, top_k=self.top_k)

        if self.verbose:
            print(f"\n  [Retriever] top-{self.top_k} series for this question:")
            relevant = retriever.retrieve(question, top_k=self.top_k)
            for s in relevant:
                print(f"    [{s['similarity']:.3f}] {s['SERIES']}: {s['INDICATOR']}")

        messages = [
            {
                "role": "system",
                "content": f"You are an economic data assistant with access to FRED API. {guide}"
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        try:
            result = self.call_llm(messages)
            
            # print(result)

            if "message" not in result:
                return {
                    "success": False,
                    "error": "No message in LLM response",
                    "raw_response": result
                }
            
            assistant_message = result["message"]
            
            if "tool_calls" not in assistant_message:
                return {
                    "success": True,
                    "tool_calls": [],
                    "direct_answer": assistant_message.get("content", ""),
                    "raw_response": result
                }
            
            extracted_calls = []
            for tool_call in assistant_message["tool_calls"]:
                args = tool_call["function"]["arguments"]

                start_date, end_date = self._resolve_tool_call_dates(args)

                extracted_calls.append({
                    "tool_call_id": tool_call.get("id", f"call_{len(extracted_calls)}"),  # for matching tool call responds with tool calls
                    "series_id": args.get("series_id", "").strip(),
                    "start_date": start_date,
                    "end_date": end_date
                })
            
            return {
                "success": True,
                "tool_calls": extracted_calls,
                "raw_response": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_response": None
            }
    
    def execute_tool_calls(self, tool_calls, use_fallback=False):
        """
        execute tool calls and return results
        
        Args:
            tool_calls: list of dict with {series_id, start_date, end_date}
            use_fallback: whether use fall back or not
        
        Returns:
            list of dict with API results
        """
        results = []
        # use compact summary mode for multiple tool calls to save tokens
        use_compact = len(tool_calls) > 2
    
        for idx, call in enumerate(tool_calls):
            series_id = call["series_id"]
            start_date = call["start_date"] or (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
            end_date = call["end_date"] or datetime.today().strftime("%Y-%m-%d")
            
            if not series_id:
                results.append({
                    "success": False,
                    "error": "No series_id provided",
                    "tool_call_id": call.get("tool_call_id", f"call_{idx}")
                })
                continue
            
            if self.verbose:
                print(f"  Tool call {idx + 1}: {series_id} ({start_date} to {end_date})")
            
            if use_fallback:
                api_result = call_fred_api_with_fallback(
                    series_id, start_date, end_date, 
                    compact_mode=use_compact  #
                )
            else:
                api_result = call_fred_api(
                    series_id, start_date, end_date,
                    compact_mode=use_compact  #
                )

                # # plot data
                # def plot_line(observations, title=None):
                #     # sort by date to ensure chronological order from past to present
                #     dates = [datetime.strptime(obs["date"], "%Y-%m-%d") for obs in observations]
                #     values = [float(obs["value"]) for obs in observations]
                #     plt.figure(figsize=(12, 6))
                #     plt.plot(dates, values, color="r")
                #     if title is not None:
                #         plt.title(title)
                #     plt.xlabel("Date", fontsize=10)
                #     plt.ylabel("Value", fontsize=10)
                #     plt.xticks(rotation=45, ha="right")  # rotate x-axis labels for better readability
                #     plt.legend()
                #     plt.grid(True, alpha=0.3) # alpha adjusts transparency
                #     plt.tight_layout()
                #     plt.show()

                # plot_line(api_result["data"], title=f"{api_result["indicator_name"]}")
            
            api_result["tool_call_id"] = call.get("tool_call_id", f"call_{idx}")
            results.append(api_result)
        
        return results
    
    def process_question(self, question):
        """
        run complete process: tool calls -> execution -> final answer generation
        
        Returns:
            dict: {
                "question": str,
                "tool_calls": list,
                "api_results": list,
                "final_answer": str,
                "execution_time": float,
                "success": bool
            }
        """
        if self.verbose:
            print(f"\n{"="*60}")
            print(f"Question: {question}")
            print("="*60)
        
        start_time = time.time()
        
        # step 1: parse tool calls
        if self.verbose:
            print("\nstep 1: Extracting tool calls from LLM...")
        
        extraction = self.extract_tool_calls(question)
        
        if not extraction["success"]:
            return {
                "question": question,
                "success": False,
                "error": extraction.get("error"),
                "execution_time": time.time() - start_time
            }
        
        tool_calls = extraction["tool_calls"]
        
        if not tool_calls:
            if self.verbose:
                print("\nNo tool calls needed. Direct answer:")
                print(extraction.get("direct_answer", "No response"))
            
            return {
                "question": question,
                "success": True,
                "tool_calls": [],
                "api_results": [],
                "final_answer": extraction.get("direct_answer", ""),
                "execution_time": time.time() - start_time
            }
        
        if self.verbose:
            print(f"\nExtracted {len(tool_calls)} tool call(s):")
            for i, call in enumerate(tool_calls):
                print(f"  {i+1}. series_id={call['series_id']}, "
                      f"dates={call["start_date"]} to {call["end_date"]}")
        
        # step 2: run tool calls
        if self.verbose:
            print("\nstep 2: Executing tool calls with auto-fallback...")
        
        api_results = self.execute_tool_calls(tool_calls, use_fallback=False)
        
        # step 3: generate final answer
        if self.verbose:
            print("\nstep 3: Generating final answer...")
        
        messages = [
            {
                "role": "system",
                "content": f"You are an economic data assistant with access to FRED API. Today is {datetime.today().strftime('%Y-%m-%d')}."
            }
        ]

        if self.few_shot:
            messages.extend(build_few_shot_messages())
        
        messages.extend([
            {
                "role": "user",
                "content": question
            },
            extraction["raw_response"]["message"]
        ])
        
        # add tool responses
        for idx, result in enumerate(api_results):
            if result["success"]:
                tool_result = {
                    "series_id": result.get("series_id", ""),
                    "indicator": result.get("indicator_name", ""),
                    "analysis": result.get("analysis", {})
                }
            else:
                tool_result = {
                    "error": result.get("error", "Unknown error")
                }

            if idx == len(api_results) - 1:
                series_list = ", ".join(r["series_id"] for r in api_results if r["success"])
                tool_result["reminder"] = (
                    f"IMPORTANT: You must analyze ALL {len(api_results)} indicators "
                    f"including {series_list}. Do not skip any."
                )

            messages.append({
                "role": "tool",
                "tool_call_id": result.get("tool_call_id", ""),
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

        final_result = self.call_llm(messages)

        final_answer = final_result.get("message", {}).get("content", "No response generated")
        
        execution_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{"="*60}")
            print("Final answer:")
            print(f"{"="*60}")
            print(final_answer)
            print(f"\nExecution time: {execution_time:.2f}s")
            print(f"{"="*60}\n")
        
        return {
            "question": question,
            "success": True,
            "tool_calls": tool_calls,
            "api_results": api_results,
            "final_answer": final_answer,
            "execution_time": execution_time
        }


def process_question(question, model="llama3.2", verbose=True, few_shot=False):

    agent = FredLLMAgent(model=model, verbose=verbose, few_shot=few_shot)

    return agent.process_question(question)


if __name__ == "__main__":
    import pandas as pd

    with open("data/QA_test.json", encoding="utf-8") as f:
        file = json.load(f)

    # agent = FredLLMAgent(model="llama3.2", verbose=False)
    # agent = FredLLMAgent(model="llama-finetuned-v2", verbose=False)
    # agent = FredLLMAgent(model="llama-finetuned-v3", verbose=False)
    # agent = FredLLMAgent(model="llama-finetuned-v4", verbose=False)
    # agent = FredLLMAgent(model="llama-finetuned-v5", verbose=False)
    agent = FredLLMAgent(model="llama-finetuned-v6", verbose=False)
    
    results = []
    for idx, question in enumerate(file):
        question_id = question["question_id"]
        print(f"Question {idx + 1}: {question_id}")
        try:
            result = agent.process_question(question["question"])
            # result = agent.process_question(question)
        except Exception as e:
            print(f"  ERROR on {question_id}: {e}, skipping...")
            result = {
                "question": question["question"],
                "success": False,
                "error": str(e)
            }
        results.append(result)

    filepath = "files/llama3.2/QA_test_llama_api_semantic_retriever_finetuned6.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults exported to {filepath}")