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
import json
from datetime import datetime, timedelta
import time

client = OpenAI(api_key=gpt_key)
MODEL = "gpt-4o-mini"

retriever = SeriesRetriever()

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

    def extract_tool_calls(self, question):
        """
        extract tool calls from LLM response

        Returns:
            dict: {
                "success": bool,
                "tool_calls": list of dict with {series_id, start_date, end_date},
                "raw_response": response object,
                "error": str (if failed)
            }
        """
        guide = build_indicator_guide(question, top_k=self.top_k)

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

    def execute_tool_calls(self, tool_calls, use_fallback=True):
        """execute tool calls and return results"""
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

    def process_question(self, question):
        """
        full pipeline: extract tool calls -> execute -> generate final answer

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

        # step 1: extract tool calls
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
                print(f"  {i+1}. series_id={call["series_id"]}, "
                      f"dates={call["start_date"]} to {call["end_date"]}")

        # step 2: execute tool calls
        if self.verbose:
            print("\nStep 2: Executing tool calls with auto-fallback...")

        api_results = self.execute_tool_calls(tool_calls, use_fallback=True)

        # step 3: build messages with tool results and generate final answer
        if self.verbose:
            print("\nStep 3: Generating final answer...")

        raw_msg = extraction["raw_response"].choices[0].message

        messages = [
            {
                "role": "system",
                "content": f"You are a US economic data assistant with access to FRED API. Today is {datetime.today().strftime('%Y-%m-%d')}."
            },
            {
                "role": "user",
                "content": question
            },
            # assistant message with tool_calls (required by OpenAI)
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

    with open('data/QA.json', encoding='utf-8') as f:
        file = json.load(f)

    # file = [
    #     # "How did unemployment and inflation change in 2024?",
    #     # "What's the trade balance trend between goods and services over the past 2 years?"
    #     # "What's the date today?",
    #       "Show me GDP data for Q1 2024"
    # ]

    # agent = FredLLMAgent(model="llama-finetuned-v1", verbose=True)

    agent = OpenAIFredAgent(verbose=True)

    results = []
    for idx, question in enumerate(file):
        question_id = question['question_id']
        if question_id in [17, 73, 14.3, 17.1, 17.2, 17.3, 61.2, 73.1, 73.2, 73.3, 78.2, 78.3]:  # not sure: 81.1
            print(f"Question {idx + 1}: {question_id}")
            result = agent.process_question(question['question'])
            results.append(result)

    filepath = 'QA_gpt_fix.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults exported to {filepath}")