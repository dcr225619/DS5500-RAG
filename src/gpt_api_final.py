from openai import OpenAI
from gpt_key import gpt_key
from fred_api import load_indicator_metadata, call_fred_api
from series_retriever import SeriesRetriever
from llama_api import (
    TOOLS,
    fix_date_parameters,
    call_fred_api_with_fallback
)
from llama_api_semantic_retriever import GUIDE_SUFFIX, build_indicator_guide
from date_parser import parse_date_range
import json
from datetime import datetime, timedelta
import time
import re

client = OpenAI(api_key=gpt_key)
MODEL = "gpt-4o-mini"

retriever = SeriesRetriever()
VALID_SERIES = retriever.get_all_series_ids()

def convert_tools_to_openai_format(tools):
    """convert Ollama tool format to OpenAI tool format"""
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        })
    return openai_tools

OPENAI_TOOLS = convert_tools_to_openai_format(TOOLS)

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

def is_relative_date(value: str) -> bool:
    """Return True if *value* looks like a relative offset rather than YYYY-MM-DD."""
    return resolve_relative_date(value) is not None

class OpenAIFredAgent:
    def __init__(self, model=MODEL, verbose=True, top_k=5):
        self.model = model
        self.verbose = verbose
        self.top_k = top_k

    def call_llm(self, messages, use_tools=True):
        """call OpenAI API"""
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if use_tools:
            kwargs["tools"] = OPENAI_TOOLS
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        return response

    def validate_tool_calls(self, tool_calls, question, max_retries=1, _depth=0):
        """
        Check B: validate that all series_ids exist in the known FRED series list.
        Re-prompts the LLM with a correction hint if invalid series are found.

        Args:
            tool_calls: list of dict with {series_id, start_date, end_date}
            question: original user question
            max_retries: maximum number of correction attempts
            _depth: internal recursion depth counter

        Returns:
            list of validated tool_calls
        """
        invalid = [c for c in tool_calls if c["series_id"] not in VALID_SERIES]
        if not invalid or _depth >= max_retries:
            return tool_calls

        if self.verbose:
            print(f"  [Check B] Invalid series: {[c['series_id'] for c in invalid]}, re-prompting...")

        correction_hint = (
            f"The following series IDs do not exist: {[c['series_id'] for c in invalid]}. "
            f"Please use only series from the provided list."
        )

        extraction = self.extract_tool_calls(
            question + f"\n\n[Correction hint: {correction_hint}]"
        )

        return self.validate_tool_calls(
            extraction.get("tool_calls", tool_calls),
            question,
            max_retries,
            _depth + 1
        )

    def extract_tool_calls(self, question, min_similarity=0.3):
        """
        Extract tool calls from LLM response.
        Includes Check A (relevance gate) and Check B (series_id validation).

        Returns:
            dict: {
                "success": bool,
                "tool_calls": list of dict with {series_id, start_date, end_date},
                "raw_response": response object,
                "direct_answer": str (if no tool calls needed),
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

        # dynamically retrieve top-k relevant series for this question
        guide = build_indicator_guide(question, top_k=self.top_k)

        if self.verbose:
            print(f"\n  [Retriever] top-{self.top_k} series for this question:")
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
            response = self.call_llm(messages, use_tools=True)
            msg = response.choices[0].message

            # no tool calls -> direct answer
            if not msg.tool_calls:
                return {
                    "success": True,
                    "tool_calls": [],
                    "direct_answer": msg.content or "",
                    "raw_response": response
                }

            extracted_calls = []
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)

                start_date = args.get("start_date", "")
                end_date = args.get("end_date", "")
                start_date, end_date = fix_date_parameters(start_date, end_date)

                extracted_calls.append({
                    "tool_call_id": tc.id,
                    "series_id": args.get("series_id", "").strip(),
                    "start_date": start_date,
                    "end_date": end_date
                })

            # Check B: validate that series_ids are in the known list
            extracted_calls = self.validate_tool_calls(extracted_calls, question)

            return {
                "success": True,
                "tool_calls": extracted_calls,
                "raw_response": response
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_response": None
            }

    def execute_tool_calls(self, tool_calls, use_fallback=False):
        """
        Execute tool calls and return FRED API results.

        Args:
            tool_calls: list of dict with {series_id, start_date, end_date}
            use_fallback: whether to use date-fallback retry on empty results

        Returns:
            list of dict with API results
        """
        results = []
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
        Check C: ask the LLM to verify that the final answer covers all retrieved
        series and actually addresses the original question.

        Args:
            question: original user question
            final_answer: the generated answer string
            api_results: list of API result dicts

        Returns:
            dict: {
                "complete": bool,
                "missing_series": list of str,
                "question_addressed": bool,
                "gap": str
            }
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

        check_response = self.call_llm(
            [{"role": "user", "content": verification_prompt}],
            use_tools=False
        )

        try:
            content = check_response.choices[0].message.content or "{}"
            # strip possible markdown code fences (only for gpt model)
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception:
            return {"complete": True, "missing": [], "question_addressed": True, "gap": ""}

    def process_question(self, question, max_self_check_loop=1):
        """
        Full pipeline: extract tool calls -> execute -> generate final answer -> self-check.

        Args:
            question: user question string
            max_self_check_loop: max Check C iterations when tool calls were made

        Returns:
            dict: {
                "question", "tool_calls", "api_results",
                "final_answer", "execution_time", "success"
            }
        """
        if self.verbose:
            print(f"\n{"="*60}")
            print(f"Question: {question}")
            print("="*60)

        start_time = time.time()

        # step 1: extract tool calls (includes Check A and Check B)
        if self.verbose:
            print("\nStep 1: Extracting tool calls from LLM...")

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

        # step 2: execute tool calls with auto-fallback
        if self.verbose:
            print("\nStep 2: Executing tool calls with auto-fallback...")

        api_results = self.execute_tool_calls(tool_calls, use_fallback=True)

        # step 3: build conversation with tool results and generate final answer
        if self.verbose:
            print("\nStep 3: Generating final answer...")

        raw_msg = extraction["raw_response"].choices[0].message

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a US economic data assistant with access to FRED API. "
                    f"Today is {datetime.today().strftime('%Y-%m-%d')}."
                )
            },
            {
                "role": "user",
                "content": question
            },
            # assistant message with tool_calls (required by OpenAI for multi-turn tool use)
            raw_msg
        ]

        # append tool results
        for idx, result in enumerate(api_results):
            if result["success"]:
                tool_result = {
                    "series_id": result.get("series_id", ""),
                    "indicator": result.get("indicator_name", ""),
                    "analysis": result.get("analysis", {})
                }
            else:
                tool_result = {"error": result.get("error", "Unknown error")}

            if idx == len(api_results) - 1:
                series_list = ", ".join(r["series_id"] for r in api_results if r["success"])
                tool_result["reminder"] = (
                    f"IMPORTANT: You must analyze ALL {len(api_results)} indicators "
                    f"including {series_list}. Do not skip any."
                )

            messages.append({
                "role": "tool",
                "tool_call_id": result.get("tool_call_id", f"call_{idx}"),
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

        final_response = self.call_llm(messages, use_tools=False)
        final_answer = final_response.choices[0].message.content or "No response generated"

        # Check C: self-check completeness and question relevance
        if len(tool_calls) > 2:
            for _ in range(max_self_check_loop):
                check = self.validate_final_answer_completeness(question, final_answer, api_results)

                if check.get("complete") and check.get("question_addressed"):
                    break

                if self.verbose:
                    print(f"    [Check C] Incomplete final answer. Regenerating...")

                missing_calls, gap_hint = '', ''
                if not check.get("complete"):
                    missing_calls = f"Missing series: {', '.join(check.get('missing_series', []))}\n"
                if not check.get("question_addressed"):
                    gap_hint = f"Your answer is incomplete. {check.get('gap', '')}\n"

                messages.append({
                    "role": "user",
                    "content": (
                        missing_calls + gap_hint +
                        f"Please revise your answer to fully address the original question: {question}"
                    )
                })

                temp = self.call_llm(messages, use_tools=False)
                final_answer = temp.choices[0].message.content or "No response generated"

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
    agent = OpenAIFredAgent(model=MODEL, verbose=verbose)
    return agent.process_question(question)


if __name__ == "__main__":
    import pandas as pd
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, '../data/QA_test.json'), encoding='utf-8') as f:
        file = json.load(f)

    agent = OpenAIFredAgent(verbose=True)

    results = []
    for idx, question in enumerate(file):
        question_id = question['question_id']
        print(f"Question {question_id}:")
        result = agent.process_question(question['question'])
        results.append(result)

    filepath = '../data/QA_finetune3.json'
    with open(os.path.join(base_dir, filepath), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults exported to {filepath}")