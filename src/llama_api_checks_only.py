"""
llama_api_checks_only.py
Variant: - date_parser   |  - semantic retrieval  |  + self-checks (A / B / C)
"""

import requests
import json
import re
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from fred_api import call_fred_api
from series_retriever import SeriesRetriever
from few_shot_examples import build_few_shot_messages

OLLAMA_URL = "http://localhost:11434/api/chat"

# ── Indicator guide (full static list) ────────────────────────────────────────
with open("../files/indicator_guide_compact.txt", encoding="utf-8") as f:
    indicator_mapping = f.read()

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

INDICATOR_GUIDE_COMP = indicator_mapping + GUIDE_SUFFIX

# ── Valid series IDs for Check B ──────────────────────────────────────────────
retriever   = SeriesRetriever()
VALID_SERIES = retriever.get_all_series_ids()

TOOLS = [
    {
        "name": "get_fred_data",
        "description": "Gain economic data from FRED API, for example GDP, unemployment rate, CPI etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "series_id": {
                    "type": "string",
                    "description": "FRED data series ID, e.g. GDP, UNRATE, CPIAUCSL"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format."
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format."
                }
            },
            "required": ["series_id", "start_date", "end_date"]
        }
    }
]

# ── Date utilities (same as llama_api.py, no date_parser dependency) ──────────
_REL_PATTERN = re.compile(
    r"""^
    (?P<sign>[-+])?
    (?:(?P<years>\d+)\s*y)?
    (?:(?P<months>\d+)\s*m)?
    (?:(?P<days>\d+)\s*d)?
    $""",
    re.IGNORECASE | re.VERBOSE,
)
_TODAY_ALIASES = {"today", "now", "current", "present"}


def resolve_relative_date(value: str, reference: datetime | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    clean = value.strip().lower()
    if clean in _TODAY_ALIASES:
        return (reference or datetime.today()).strftime("%Y-%m-%d")
    try:
        datetime.strptime(clean, "%Y-%m-%d")
        return None
    except ValueError:
        pass
    sign, body = (-1, clean[1:]) if clean.startswith("-") else \
                 (1,  clean[1:]) if clean.startswith("+") else (-1, clean)
    m = _REL_PATTERN.match(body)
    if not m or not any(m.group(k) for k in ("years", "months", "days")):
        return None
    ref = reference or datetime.today()
    delta = relativedelta(
        years=sign  * int(m.group("years")  or 0),
        months=sign * int(m.group("months") or 0),
        days=sign   * int(m.group("days")   or 0),
    )
    return min(ref + delta, ref).strftime("%Y-%m-%d")


def fix_date_parameters(start_date, end_date):
    today = datetime.today()
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
    except Exception:
        print("  Invalid date format, defaulting to past 1 year")
        return (today - timedelta(days=365)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
    if end_dt > today:
        print(f"  End date {end_date} is in the future, capping to today")
        end_dt = today
    if start_dt.date() == end_dt.date():
        print("  Detected same-day query, expanding to 1 year")
        start_dt = end_dt - timedelta(days=365)
    elif start_dt > end_dt:
        print("  Start date later than end date, adjusting to 1 year ago")
        start_dt = end_dt - timedelta(days=365)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


# ── Agent ──────────────────────────────────────────────────────────────────────
class FredLLMAgent:
    def __init__(self, model="llama3.2", api_url=OLLAMA_URL, verbose=True, few_shot=False):
        self.model    = model
        self.api_url  = api_url
        self.verbose  = verbose
        self.few_shot = few_shot

    def call_llm(self, messages):
        payload = {"model": self.model, "messages": messages, "tools": TOOLS, "stream": False}
        return requests.post(self.api_url, json=payload, timeout=600).json()

    # ── date resolution (LLM absolute > LLM relative, NO date_parser) ─────────
    def _resolve_tool_call_dates(self, args):
        raw_start = (args.get("start_date") or args.get("start", "")).strip()
        raw_end   = (args.get("end_date")   or args.get("end",   "")).strip()

        resolved_start = resolve_relative_date(raw_start)
        resolved_end   = resolve_relative_date(raw_end)
        start_date = resolved_start or raw_start
        end_date   = resolved_end   or raw_end

        if self.verbose:
            if resolved_start:
                print(f"  [DateResolver] Relative start '{raw_start}' → '{start_date}'")
            if resolved_end:
                print(f"  [DateResolver] Relative end   '{raw_end}'   → '{end_date}'")

        return fix_date_parameters(start_date, end_date)

    # ── Check B: validate series IDs, re-prompt if invalid ────────────────────
    def validate_tool_calls(self, tool_calls, question, max_retries=1, _depth=0):
        invalid = [c for c in tool_calls if c["series_id"] not in VALID_SERIES]
        if not invalid or _depth >= max_retries:
            return tool_calls

        if self.verbose:
            print(f"  [Check B] Invalid series: {[c['series_id'] for c in invalid]}, re-prompting...")

        hint = (f"The following series IDs do not exist: {[c['series_id'] for c in invalid]}. "
                "Please use only series from the provided list.")
        extraction = self.extract_tool_calls(question + f"\n\n[Correction hint: {hint}]")
        return self.validate_tool_calls(extraction.get("tool_calls", tool_calls), question,
                                        max_retries, _depth + 1)

    # ── step 1: extract tool calls ─────────────────────────────────────────────
    def extract_tool_calls(self, question):
        # No semantic retrieval — full guide is always passed
        messages = [
            {"role": "system", "content": f"You are an economic data assistant with access to FRED API. {INDICATOR_GUIDE_COMP}"},
            {"role": "user",   "content": question}
        ]
        try:
            result        = self.call_llm(messages)
            assistant_msg = result.get("message", {})

            if "message" not in result:
                return {"success": False, "error": "No message in LLM response", "raw_response": result}

            if "tool_calls" not in assistant_msg:
                return {
                    "success": True, "tool_calls": [],
                    "direct_answer": assistant_msg.get("content", ""),
                    "raw_response": result
                }

            extracted = []
            for tc in assistant_msg["tool_calls"]:
                args  = tc["function"]["arguments"]
                start, end = self._resolve_tool_call_dates(args)
                extracted.append({
                    "tool_call_id": tc.get("id", f"call_{len(extracted)}"),
                    "series_id":    args.get("series_id", "").strip(),
                    "start_date":   start,
                    "end_date":     end,
                })

            # Check B: validate series IDs
            extracted = self.validate_tool_calls(extracted, question)

            return {"success": True, "tool_calls": extracted, "raw_response": result}

        except Exception as e:
            return {"success": False, "error": str(e), "raw_response": None}

    # ── step 2: execute tool calls ─────────────────────────────────────────────
    def execute_tool_calls(self, tool_calls):
        results     = []
        use_compact = len(tool_calls) > 2
        for idx, call in enumerate(tool_calls):
            sid   = call["series_id"]
            start = call["start_date"] or (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
            end   = call["end_date"]   or datetime.today().strftime("%Y-%m-%d")

            if not sid:
                results.append({"success": False, "error": "No series_id provided",
                                 "tool_call_id": call.get("tool_call_id", f"call_{idx}")})
                continue

            if self.verbose:
                print(f"  Tool call {idx+1}: {sid} ({start} to {end})")

            api_result = call_fred_api(sid, start, end, compact_mode=use_compact)
            api_result["tool_call_id"] = call.get("tool_call_id", f"call_{idx}")
            results.append(api_result)
        return results

    # ── Check C: validate completeness of final answer ─────────────────────────
    def validate_final_answer_completeness(self, question, final_answer, api_results):
        series_used = [r["series_id"] for r in api_results if r["success"]]
        prompt = f"""
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
        try:
            content = self.call_llm([{"role": "user", "content": prompt}])\
                          .get("message", {}).get("content", "{}")
            return json.loads(content)
        except Exception:
            return {"complete": True, "missing_series": [], "question_addressed": True, "gap": ""}

    # ── main entry point ───────────────────────────────────────────────────────
    def process_question(self, question, max_self_check_loop=1):
        if self.verbose:
            print(f"\n{'='*60}\nQuestion: {question}\n{'='*60}")

        start_time = time.time()

        if self.verbose:
            print("\nstep 1: Extracting tool calls from LLM...")
        extraction = self.extract_tool_calls(question)

        if not extraction["success"]:
            return {"question": question, "success": False,
                    "error": extraction.get("error"),
                    "execution_time": time.time() - start_time}

        tool_calls = extraction["tool_calls"]

        if not tool_calls:
            if self.verbose:
                print("\nNo tool calls needed. Direct answer:")
                print(extraction.get("direct_answer", "No response"))
            return {"question": question, "success": True, "tool_calls": [],
                    "api_results": [], "final_answer": extraction.get("direct_answer", ""),
                    "execution_time": time.time() - start_time}

        if self.verbose:
            print(f"\nExtracted {len(tool_calls)} tool call(s):")
            for i, c in enumerate(tool_calls):
                print(f"  {i+1}. series_id={c['series_id']}, dates={c['start_date']} to {c['end_date']}")

        if self.verbose:
            print("\nstep 2: Executing tool calls...")
        api_results = self.execute_tool_calls(tool_calls)

        if self.verbose:
            print("\nstep 3: Generating final answer...")

        messages = [{"role": "system", "content": f"You are an economic data assistant with access to FRED API. Today is {datetime.today().strftime('%Y-%m-%d')}."}]
        if self.few_shot:
            messages.extend(build_few_shot_messages())
        messages.extend([
            {"role": "user", "content": question},
            extraction["raw_response"]["message"]
        ])

        for idx, result in enumerate(api_results):
            tool_result = ({"series_id": result.get("series_id", ""),
                            "indicator": result.get("indicator_name", ""),
                            "analysis":  result.get("analysis", {})}
                           if result["success"] else {"error": result.get("error", "Unknown error")})
            if idx == len(api_results) - 1:
                series_list = ", ".join(r["series_id"] for r in api_results if r["success"])
                tool_result["reminder"] = (
                    f"IMPORTANT: You must analyze ALL {len(api_results)} indicators "
                    f"including {series_list}. Do not skip any."
                )
            messages.append({"role": "tool",
                             "tool_call_id": result.get("tool_call_id", ""),
                             "content": json.dumps(tool_result, ensure_ascii=False)})

        final_answer = self.call_llm(messages).get("message", {}).get("content", "No response generated")

        # Check C: self-check completeness when many series are involved
        if len(tool_calls) > 2:
            for _ in range(max_self_check_loop):
                check = self.validate_final_answer_completeness(question, final_answer, api_results)
                if check.get("complete") and check.get("question_addressed"):
                    break

                if self.verbose:
                    print("    [Check C] Incomplete final answer. Regenerating...")

                missing_calls = (f"Missing series: {', '.join(check.get('missing_series', []))}\n"
                                 if not check.get("complete") else "")
                gap_hint      = (f"Your answer is incomplete. {check.get('gap', '')}\n"
                                 if not check.get("question_addressed") else "")

                messages.append({"role": "user",
                                  "content": missing_calls + gap_hint +
                                             f"Please revise your answer to fully address the original question: {question}"})
                final_answer = self.call_llm(messages).get("message", {}).get("content", "No response generated")

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"\n{'='*60}\nFinal answer:\n{'='*60}\n{final_answer}")
            print(f"\nExecution time: {execution_time:.2f}s\n{'='*60}\n")

        return {"question": question, "success": True, "tool_calls": tool_calls,
                "api_results": api_results, "final_answer": final_answer,
                "execution_time": execution_time}


def process_question(question, model="llama3.2", verbose=True, few_shot=False):
    return FredLLMAgent(model=model, verbose=verbose, few_shot=few_shot).process_question(question)


if __name__ == "__main__":
    with open("../data/QA_test.json", encoding="utf-8") as f:
        file = json.load(f)

    agent = FredLLMAgent(model="llama3.2", verbose=False)

    results = []
    for idx, question in enumerate(file):
        question_id = question["question_id"]
        print(f"Question {idx+1}: {question_id}")
        try:
            result = agent.process_question(question["question"])
        except Exception as e:
            print(f"  ERROR on {question_id}: {e}, skipping...")
            result = {"question": question["question"], "success": False, "error": str(e)}
        results.append(result)

    filepath = "../files/llama3.2/QA_test_llama_api_checks_only.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults exported to {filepath}")