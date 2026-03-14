# self-check version for llama_api.py

import requests
from fred_key import fred_key
from fred_api import load_indicator_metadata, call_fred_api
from series_retriever import SeriesRetriever
import json
from datetime import datetime, timedelta
import time

BASE_URL = "https://api.stlouisfed.org/fred/"
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
        
        response = requests.post(self.api_url, json=payload)
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

        # Check A:
        # if the maximum correlation is too low, indicating that the issue may lie outside the scope of the data
        relevant = retriever.retrieve(question, top_k=self.top_k)
        top_score = relevant[0]["similarity"] if relevant else 0

        # return warning for no data series with relevance larger than 0.3
        if top_score < min_similarity:
            if self.verbose:
                print("  [Check A] No relevant FRED series found.")

            return {
                "success": False,
                "error": f"No relevant FRED series found (best score: {top_score:.3f}). "
                        f"This question may be outside FRED's coverage."
            }

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

                start_date = args.get("start_date") or args.get("start", "")
                end_date = args.get("end_date") or args.get("end", "")
                
                # fix problematic dates
                start_date, end_date = fix_date_parameters(start_date, end_date)

                extracted_calls.append({
                    "tool_call_id": tool_call.get("id", f"call_{len(extracted_calls)}"),  # for matching tool call responds with tool calls
                    "series_id": args.get("series_id", "").strip(),
                    "start_date": start_date,
                    "end_date": end_date
                })

            # Check B:
            # make sure the series ids are available
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
                print(f"  {i+1}. series_id={call["series_id"]}, "
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
            },
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
        if len(tool_calls) > 0:
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
                    gap_hint = f"Your answer is incomplete. {check.get("gap", "")}\n"
                
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

    # with open("data/QA.json", encoding="utf-8") as f:
    #     file = json.load(f)
    
    file = [
        "How did unemployment and inflation change in 2024?",
        "What's the trade balance trend between goods and services over the past 2 years?"
        "What's the date today?",
        "Show me GDP data for Q1 2024"
    ]

    agent = FredLLMAgent(model="llama3.2", verbose=True)
    # agent = FredLLMAgent(model="llama-finetuned-v2", verbose=True)
    # agent = FredLLMAgent(model="llama-finetuned-v3", verbose=True)
    
    results = []
    for idx, question in enumerate(file):
        # question_id = question["question_id"]
        # print(f"Question {idx + 1}: {question_id}")
        # result = agent.process_question(question["question"])
        #
        result = agent.process_question(question)
        #
        results.append(result)

    # print(results)

    # filepath = "files/finetune-2/all_results_compact.json"
    # with open(filepath, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)

    # print(f"\nResults exported to {filepath}")