import requests
from fred_key import fred_key
from fred_api import load_indicator_metadata, call_fred_api
import json
from datetime import datetime, timedelta
import time

BASE_URL = 'https://api.stlouisfed.org/fred/'
OLLAMA_URL = "http://localhost:11434/api/chat"

today = datetime.today()

# indicator_guide_compact.txt, indicator_guide_by_category.txt, indicator_guide_optimized.txt, indicator_guide_with_examples.txt
with open('files/indicator_guide_compact.txt', encoding='utf-8') as f:
    indicator_mapping = f.read()

# def build_indicator_guide(output_json_path='output.json'):
#     with open(output_json_path, 'r', encoding='utf-8') as f:
#         indicators = json.load(f)
    
#     lines = []
#     current_category = None
    
#     # divided by CATEGORY
#     from itertools import groupby
#     sorted_indicators = sorted(indicators, key=lambda x: (x['CATEGORY'], x['SUB-CATEGORY']))
    
#     for item in sorted_indicators:
#         if item['CATEGORY'] != current_category:
#             current_category = item['CATEGORY']
#             lines.append(f"\n## {current_category}")
        
#         period_map = {'M': 'Monthly', 'Q': 'Quarterly', 'W': 'Weekly', 'D': 'Daily', 'A': 'Annual'}
#         period = period_map.get(item['PERIOD'], item['PERIOD'])
        
#         lines.append(
#             f"- {item['SERIES']}: {item['INDICATOR']} | {item['UNITS']} | {period}"
#         )
    
#     return "\n".join(lines)

# indicator_mapping = build_indicator_guide()

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
                    "description": "Start date in YYYY-MM-DD format. For 'current/recent' queries, use 2 years ago. For specific periods, use the exact start date."
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
    """
    fix date parameters to ensure data retrieval
    
    Args:
        series_id: FRED series ID
        start_date
        end_date
    
    Returns:
        tuple: (fixed_start_date, fixed_end_date)
    """
    today = datetime.today()
    
    # 1. date format check
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except:
        start_dt = today - timedelta(days=365)  # 1 years ago
        end_dt = today
    
    # 2. both start and end are the same day
    if start_dt.date() == end_dt.date():
        print(f"  Detected same-day query, expanding to 1 years")
        start_dt = today - timedelta(days=365)
        end_dt = today
    
    # 3. start > end
    elif start_dt > end_dt:
        print(f"  Start date later than end date, adjusting to 1 years ago")
        start_dt = end_dt - timedelta(days=365)
    
    return start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')


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
        
        if result['success'] and result.get('data'):
            if attempt > 0:
                print(f"  Retry {attempt} succeeded with date range: {start_date} to {end_date}")
            return result
        
        # fall back
        if attempt < max_retries - 1:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                start_dt = start_dt - timedelta(days=365)
                start_date = start_dt.strftime('%Y-%m-%d')
                print(f"  No data found, retrying with end_date: {start_date}")
            except:
                break
    
    return result


class FredLLMAgent:
    def __init__(self, model="llama3.2", api_url=OLLAMA_URL, verbose=True):
        self.model = model
        self.api_url = api_url
        self.verbose = verbose  # print process or not
        
    def call_llm(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS,
            "stream": False
        }
        
        response = requests.post(self.api_url, json=payload)
        return response.json()
    
    def extract_tool_calls(self, question):
        """
        extract tool calls without execution
        
        Returns:
            dict: {
                'success': bool,
                'tool_calls': list of dict with {series_id, start_date, end_date},
                'raw_response': dict,
                'error': str (if failed)
            }
        """
        messages = [
            {
                "role": "system",
                "content": f"You are an economic data assistant with access to FRED API. {INDICATOR_GUIDE}"
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
                    'success': False,
                    'error': 'No message in LLM response',
                    'raw_response': result
                }
            
            assistant_message = result["message"]
            
            if "tool_calls" not in assistant_message:
                return {
                    'success': True,
                    'tool_calls': [],
                    'direct_answer': assistant_message.get('content', ''),
                    'raw_response': result
                }
            
            extracted_calls = []
            for tool_call in assistant_message["tool_calls"]:
                args = tool_call["function"]["arguments"]

                start_date = args.get('start_date') or args.get('start', '')
                end_date = args.get('end_date') or args.get('end', '')
                
                # fix problematic dates
                start_date, end_date = fix_date_parameters(start_date, end_date)

                extracted_calls.append({
                    'tool_call_id': tool_call.get('id', f'call_{len(extracted_calls)}'),  # for matching tool call responds with tool calls
                    'series_id': args.get('series_id', '').strip(),
                    'start_date': start_date,
                    'end_date': end_date
                })
            
            return {
                'success': True,
                'tool_calls': extracted_calls,
                'raw_response': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'raw_response': None
            }
    
    def execute_tool_calls(self, tool_calls, use_fallback=True):
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
            series_id = call['series_id']
            start_date = call['start_date'] or (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
            end_date = call['end_date'] or datetime.today().strftime('%Y-%m-%d')
            
            if not series_id:
                results.append({
                    'success': False,
                    'error': 'No series_id provided',
                    'tool_call_id': call.get('tool_call_id', f'call_{idx}')
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
                #     dates = [datetime.strptime(obs['date'], "%Y-%m-%d") for obs in observations]
                #     values = [float(obs['value']) for obs in observations]
                #     plt.figure(figsize=(12, 6))
                #     plt.plot(dates, values, color='r')
                #     if title is not None:
                #         plt.title(title)
                #     plt.xlabel('Date', fontsize=10)
                #     plt.ylabel('Value', fontsize=10)
                #     plt.xticks(rotation=45, ha='right')  # rotate x-axis labels for better readability
                #     plt.legend()
                #     plt.grid(True, alpha=0.3) # alpha adjusts transparency
                #     plt.tight_layout()
                #     plt.show()

                # plot_line(api_result['data'], title=f'{api_result['indicator_name']}')
            
            api_result['tool_call_id'] = call.get('tool_call_id', f'call_{idx}')
            results.append(api_result)
        
        return results
    
    def process_question(self, question):
        """
        run complete process: tool calls -> execution -> final answer generation
        
        Returns:
            dict: {
                'question': str,
                'tool_calls': list,
                'api_results': list,
                'final_answer': str,
                'execution_time': float,
                'success': bool
            }
        """
        if self.verbose:
            print(f'\n{"="*60}')
            print(f'Question: {question}')
            print("="*60)
        
        start_time = time.time()
        
        # step 1: parse tool calls
        if self.verbose:
            print("\nstep 1: Extracting tool calls from LLM...")
        
        extraction = self.extract_tool_calls(question)
        
        if not extraction['success']:
            return {
                'question': question,
                'success': False,
                'error': extraction.get('error'),
                'execution_time': time.time() - start_time
            }
        
        tool_calls = extraction['tool_calls']
        
        if not tool_calls:
            if self.verbose:
                print("\nNo tool calls needed. Direct answer:")
                print(extraction.get('direct_answer', 'No response'))
            
            return {
                'question': question,
                'success': True,
                'tool_calls': [],
                'api_results': [],
                'final_answer': extraction.get('direct_answer', ''),
                'execution_time': time.time() - start_time
            }
        
        if self.verbose:
            print(f"\nExtracted {len(tool_calls)} tool call(s):")
            for i, call in enumerate(tool_calls):
                print(f"  {i+1}. series_id={call['series_id']}, "
                      f"dates={call['start_date']} to {call['end_date']}")
        
        # step 2: run tool calls
        if self.verbose:
            print("\nstep 2: Executing tool calls with auto-fallback...")
        
        api_results = self.execute_tool_calls(tool_calls, use_fallback=True)
        
        # step 3: generate final answer
        if self.verbose:
            print("\nstep 3: Generating final answer...")
        
        messages = [
            {
                "role": "system",
                "content": f"You are an US economic data assistant with access to FRED API. {INDICATOR_GUIDE}"
            },
            {
                "role": "user",
                "content": question
            },
            extraction['raw_response']['message']
        ]
        
        # add tool responses
        for idx, result in enumerate(api_results):
            if result['success']:
                tool_result = {
                    'series_id': result.get('series_id', ''),
                    'indicator': result.get('indicator_name', ''),
                    'analysis': result.get('analysis', {})
                }
            else:
                tool_result = {
                    'error': result.get('error', 'Unknown error')
                }

            if idx == len(api_results) - 1:
                series_list = ', '.join(r['series_id'] for r in api_results if r['success'])
                tool_result['reminder'] = (
                    f"IMPORTANT: You must analyze ALL {len(api_results)} indicators "
                    f"including {series_list}. Do not skip any."
                )

            messages.append({
                "role": "tool",
                "tool_call_id": result.get('tool_call_id', ''),
                "content": json.dumps(tool_result, ensure_ascii=False)
            })
        
        # # DYNAMIC UPDATE system prompt to make sure that the llm uses all the data and answer the question
        # series_list = ', '.join(r['series_id'] for r in api_results if r['success'])
        # 
        # messages[0]['content'] = (
        #     f"You are an economic data assistant with access to FRED API. {INDICATOR_GUIDE}\n\n"
        #     f"REQUIRED: You have received {len(api_results)} datasets ({series_list}). "
        #     f"You MUST explicitly analyze ALL of them in your response. Do not skip any."
        # )

        final_result = self.call_llm(messages)

        final_answer = final_result.get("message", {}).get("content", "No response generated")
        
        execution_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Final answer:")
            print(f"{'='*60}")
            print(final_answer)
            print(f"\nExecution time: {execution_time:.2f}s")
            print(f"{'='*60}\n")
        
        return {
            'question': question,
            'success': True,
            'tool_calls': tool_calls,
            'api_results': api_results,
            'final_answer': final_answer,
            'execution_time': execution_time
        }


def process_question(question, verbose=True):

    # original version
    # agent = FredLLMAgent(model="llama3.2", verbose=True)

    # fine-tuned-1 version
    agent = FredLLMAgent(model="llama-finetuned-v1", verbose=True)

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

    agent = FredLLMAgent(model="llama3.2", verbose=True)
    
    for idx, question in enumerate(file):
        if idx < 1:
            continue
        if idx >= 4:
            break 
        result = agent.process_question(question['question'])
        # result = agent.process_question(question)