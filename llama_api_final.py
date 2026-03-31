# self-check version for llama_api.py

import requests
import re
from fred_key import fred_key
from fred_api import load_indicator_metadata, call_fred_api
from series_retriever import SeriesRetriever
import json
from datetime import datetime, timedelta
from date_parser import parse_date_range
from dateutil.relativedelta import relativedelta
import time
from few_shot_examples import build_few_shot_messages
import re

OLLAMA_URL = "http://localhost:11434/api/chat"

retriever = SeriesRetriever()
VALID_SERIES = retriever.get_all_series_ids()

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

# ─────────────────────────────────────────────
# Relative date resolver
# ─────────────────────────────────────────────

# Patterns the model might return instead of absolute dates, e.g.
#   "-1y", "-2y", "-18m", "-6m", "-3y", "today", "now", "-1y6m" …
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

    # ── 1. "today" aliases ────────────────────────────────────────────────
    if clean in _TODAY_ALIASES:
        ref = reference or datetime.today()
        return ref.strftime("%Y-%m-%d")

    # ── 2. Already an absolute YYYY-MM-DD → not relative ──────────────────
    try:
        datetime.strptime(clean, "%Y-%m-%d")
        return None
    except ValueError:
        pass

    # ── 3. Regex match for offset patterns ────────────────────────────────
    # Detect and strip leading sign, default direction is negative (past)
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

    # Cap future dates to today
    today = reference or datetime.today()
    if result_dt > today:
        result_dt = today

    return result_dt.strftime("%Y-%m-%d")


def is_relative_date(value: str) -> bool:
    """Return True if *value* looks like a relative offset rather than YYYY-MM-DD."""
    return resolve_relative_date(value) is not None

class FredLLMAgent:
    def __init__(self, model="llama3.2", api_url=OLLAMA_URL, verbose=True, top_k=5):
        self.model = model
        self.api_url = api_url
        self.verbose = verbose  # print process or not
        self.top_k = top_k
        
    def call_llm(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS,
            "stream": False
        }
        
        response = requests.post(self.api_url, json=payload, timeout=600)
        return response.json()
    
    def validate_tool_calls(self, tool_calls, question, max_retries=1, _depth=0):
        """
        validate tool calls

        Args:
            tool_calls
            question
            max_retries: maximum times of retry
            _depth: internal count

        Returns:
            tool_calls: list of dict with {series_id, start_date, end_date}
        """
        invalid = [c for c in tool_calls if c["series_id"] not in VALID_SERIES]
        if not invalid or _depth >= max_retries:
            return tool_calls
        
        if self.verbose:
            print(f"  [Check B] Invalid series: {[c['series_id'] for c in invalid]}, re-prompting...")
        
        # re-prompting LLM, pointing out which series ids are not available
        correction_hint = f"The following series IDs do not exist: {[c['series_id'] for c in invalid]}. Please use only series from the provided list."

        # re-call extract_tool_calls()
        extraction = self.extract_tool_calls(question + f"\n\n[Correction hint: {correction_hint}]")

        return self.validate_tool_calls(extraction.get("tool_calls", tool_calls), question, max_retries, _depth + 1)
    
    def _resolve_tool_call_dates(
        self,
        args: dict,
        pre_start: str | None,
        pre_end: str | None,
    ) -> tuple[str, str]:
        """
        Determine the final (start_date, end_date) pair for a single tool call.

        Priority order
        ──────────────
        1. pre-parsed dates from date_parser  (most reliable)
        2. absolute dates already in YYYY-MM-DD coming from the LLM
        3. relative dates returned by the LLM  → resolved to absolute
        4. fallback: fix_date_parameters sanitisation

        Args:
            args      : the raw arguments dict from the LLM tool call
            pre_start : start date from date_parser (may be None)
            pre_end   : end date   from date_parser (may be None)

        Returns:
            (start_date, end_date) both in YYYY-MM-DD format
        """
        today = datetime.today()

        # ── Priority 1: date_parser result ────────────────────────────────
        if pre_start and pre_end:
            if self.verbose:
                print(f"  [DateResolver] Using date_parser result: {pre_start} → {pre_end}")
            return pre_start, pre_end

        # ── Priority 2 & 3: LLM-provided dates ────────────────────────────
        raw_start = (args.get("start_date") or args.get("start", "")).strip()
        raw_end   = (args.get("end_date")   or args.get("end",   "")).strip()

        # Try to resolve each value; resolve_relative_date returns None for
        # strings that are already absolute YYYY-MM-DD.
        resolved_start = resolve_relative_date(raw_start)
        resolved_end   = resolve_relative_date(raw_end)

        start_date = resolved_start if resolved_start else raw_start
        end_date   = resolved_end   if resolved_end   else raw_end

        if self.verbose:
            if resolved_start:
                print(f"  [DateResolver] Relative start '{raw_start}' → '{start_date}'")
            if resolved_end:
                print(f"  [DateResolver] Relative end   '{raw_end}'   → '{end_date}'")

        # ── Priority 4: sanitise whatever we ended up with ─────────────────
        start_date, end_date = fix_date_parameters(start_date, end_date)
        return start_date, end_date

    def extract_tool_calls(self, question, min_similarity=0.3):
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

        # before calling the LLM,  a semantic retriever is used to find the top-k most relevant series for the given question, 
        # and only those are passed into the prompt.
        relevant = retriever.retrieve(question, top_k=self.top_k)
        top_score = relevant[0]["similarity"] if relevant else 0

        # Check A:
        # if the maximum similarity score is too low, indicating that the question may lie outside the scope of the data
        # skip the tool call entirely and directly generate final summary
        if top_score < min_similarity:
            if self.verbose:
                print("  [Check A] No relevant FRED series found.")

            return {
                "success": False,
                "error": f"No relevant FRED series found (best score: {top_score:.3f}). "
                        f"This question may be outside FRED's coverage."
            }

        # pre-parse date range from question before calling LLM
        pre_start, pre_end = parse_date_range(question)

        if self.verbose:
            if pre_start:
                print(f"  [DateParser] Detected range: {pre_start} to {pre_end}")
            else:
                print(f"  [DateParser] No unambiguous date found, LLM will decide")

        # build dynamic indicator guide for llm
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

                start_date, end_date = self._resolve_tool_call_dates(
                    args, pre_start, pre_end
                )

                extracted_calls.append({
                    "tool_call_id": tool_call.get("id", f"call_{len(extracted_calls)}"),
                    "series_id": args.get("series_id", "").strip(),
                    "start_date": start_date,
                    "end_date": end_date
                })

            # Check B: make sure the series ids are available
            # otherwise reprompt with a correction hint and ask llm to regenerate the parameters
            extracted_calls = self.validate_tool_calls(extracted_calls, question)
            
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
                    compact_mode=use_compact
                )
            else:
                api_result = call_fred_api(
                    series_id, start_date, end_date,
                    compact_mode=use_compact
                )
            
            api_result["tool_call_id"] = call.get("tool_call_id", f"call_{idx}")
            results.append(api_result)
        
        return results
    
    def validate_final_answer_completeness(self, question, final_answer, api_results):
        """
        validate whether the final answer includes all the series called and fall back if needed

        Args:
            question
            final_answer
            api_results
        Returns:
            JSON: {{
            "complete": true/false,
            "missing_series": ["series_id that were not discussed"],
            "question_addressed": true/false,
            "gap": "one sentence describing what's missing from the answer, or empty string if complete"
            }}
        """
        series_used = [r["series_id"] for r in api_results if r["success"]]
    
        verification_prompt = f"""
        You are reviewing an economic data analysis response.

        Original question: "{question}"
        Datasets retrieved: {series_used}
        Answer provided: "{final_answer[:600]}..."

        Evaluate the answer on TWO criteria:
        1. DATA COVERAGE: Does the answer explicitly discuss all datasets listed?
        2. QUESTION RELEVANCE: Does the answer actually address what was asked in the original question?

        Reply with JSON only, no explanation:
        {{
        "complete": true/false,
        "missing_series": ["series_id that were not discussed"],
        "question_addressed": true/false,
        "gap": "one sentence describing what's missing from the answer, or empty string if complete"
        }}
        """
        check_result = self.call_llm([
            {"role": "user", "content": verification_prompt}
        ])
        
        try:
            content = check_result.get("message", {}).get("content", "{}")
            parsed = json.loads(content)
            return parsed
        except:
            return {"complete": True, "missing": [], "question_addressed": True, "gap": ""}


    def process_question(self, question, max_self_check_loop=1):
        """
        run complete process: tool calls -> execution -> final answer generation

        Args:
            question
            max_self_check_loop: the maximum self-check for final answer when len(tool_calls) > 2
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
                      f"dates={call['start_date']} to {call['end_date']}")
        
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
            },
            # *build_few_shot_messages(),  # few-shot examples
            {
                "role": "user",
                "content": question
            },
            extraction["raw_response"]["message"]
        ]
        
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
        
        # Check C:
        # self-check the completeness of the final answer when question involves more than 2 series data
        if len(tool_calls) > 2:
            for _ in range(max_self_check_loop):
                check = self.validate_final_answer_completeness(question, final_answer, api_results)
                if check.get("complete") and check.get("question_addressed"):
                    break

                if self.verbose:
                    print(f"    [Check C] Incomplete final answer. Regenerating...")

                # check if the answer covers all the retrieved data
                missing_calls, gap_hint = '', ''
                if not check.get("complete"):
                    missing_calls = f"Missing series: {', '.join(check.get('missing_series', []))}\n"
                if not check.get("question_addressed"):
                    gap_hint = f"Your answer is incomplete. {check.get('gap', '')}\n"
                
                messages.append({
                    "role": "user",
                    "content": missing_calls + gap_hint + f"Please revise your answer to fully address the original question: {question}",
                })
                temp = self.call_llm(messages)
                final_answer = temp.get("message", {}).get("content", "No response generated")

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


def process_question(question, verbose=True):

    # original version
    # agent = FredLLMAgent(model="llama3.2", verbose=True)

    # fine-tuned-1 version
    agent = FredLLMAgent(model="llama-finetuned-v1", verbose=True)

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

    filepath = "files/llama3.2/QA_test_llama_api_final_finetuned6.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults exported to {filepath}")