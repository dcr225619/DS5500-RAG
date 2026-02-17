import requests
from fred_key import fred_key
from fred_api import load_indicator_metadata, call_fred_api
import json
from datetime import datetime, timedelta
import time

BASE_URL = 'https://api.stlouisfed.org/fred/'
OLLAMA_URL = "http://localhost:11434/api/chat"

# indicator_guide_compact.txt, indicator_guide_by_category.txt, indicator_guide_optimized.txt, indicator_guide_with_examples.txt
with open('files/indicator_guide_compact.txt', encoding='utf-8') as f:
    indicator_mapping = f.read()

INDICATOR_GUIDE = indicator_mapping + f"""
Note: (M)=Monthly, (Q)=Quarterly, (W)=Weekly, (D)=Daily, (Y)=Yearly

When asked about economic data, use the get_fred_data function with the appropriate series_id.

IMPORTANT: 
1. Always specify start_date and end_date based on the user's question. Always use YYYY-MM-DD format, NOT relative dates like "-2y"
2. If no time period specified, use recent 1 year by default
3. If you only need today's or most recent data, use today - 1 year as start date and today as end date
4. Today is {datetime.today().strftime('%Y-%m-%d')}
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

def fix_date_parameters(series_id, start_date, end_date):
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
    
    # 2. both start and end are today
    if start_dt.date() == end_dt.date() == today.date():
        print(f"  Detected same-day query, expanding to 1 years")
        start_dt = today - timedelta(days=365)
        end_dt = today
    
    # 3. start > end
    elif start_dt > end_dt:
        print(f"  Start date later than end date, adjusting to 1 years ago")
        start_dt = end_dt - timedelta(days=365)
    
    return start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')


def call_fred_api_with_fallback(series_id, start_date, end_date, max_retries=1):
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
    # fix obvious date problem
    start_date, end_date = fix_date_parameters(series_id, start_date, end_date)
    
    for attempt in range(max_retries):
        result = call_fred_api(series_id, start_date, end_date)
        
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
        extract llm tool calls without executing for retrieval test
        
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
            
            if "message" not in result:
                return {
                    'success': False,
                    'error': 'No message in LLM response',
                    'raw_response': result
                }
            
            assistant_message = result["message"]
            
            # directly response if no tool call
            if "tool_calls" not in assistant_message:
                return {
                    'success': True,
                    'tool_calls': [],
                    'direct_answer': assistant_message.get('content', ''),
                    'raw_response': result
                }
            
            # parse tool calls
            extracted_calls = []
            for tool_call in assistant_message["tool_calls"]:
                args = tool_call["function"]["arguments"]
                extracted_calls.append({
                    'series_id': args.get('series_id', '').strip(),
                    'start_date': args.get('start_date') or args.get('start', ''),
                    'end_date': args.get('end_date') or args.get('end', '')
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
        
        for idx, call in enumerate(tool_calls):
            series_id = call['series_id']
            start_date = call['start_date'] or (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
            end_date = call['end_date'] or datetime.today().strftime('%Y-%m-%d')
            
            if not series_id:
                results.append({
                    'success': False,
                    'error': 'No series_id provided',
                    'tool_call_index': idx
                })
                continue
            
            if self.verbose:
                print(f"  Tool call {idx + 1}: {series_id} ({start_date} to {end_date})")
            
            if use_fallback:
                api_result = call_fred_api_with_fallback(series_id, start_date, end_date)
            else:
                api_result = call_fred_api(series_id, start_date, end_date)

            # plot data
            def plot_line(observations, title=None):
                # sort by date to ensure chronological order from past to present
                dates = [datetime.strptime(obs['date'], "%Y-%m-%d") for obs in observations]
                values = [float(obs['value']) for obs in observations]
                plt.figure(figsize=(12, 6))
                plt.plot(dates, values, color='r')
                if title is not None:
                    plt.title(title)
                plt.xlabel('Date', fontsize=10)
                plt.ylabel('Value', fontsize=10)
                plt.xticks(rotation=45, ha='right')  # rotate x-axis labels for better readability
                plt.legend()
                plt.grid(True, alpha=0.3) # alpha adjusts transparency
                plt.tight_layout()
                plt.show()

            # plot_line(api_result['data'], title=f'{api_result['indicator_name']}')
            
            api_result['tool_call_index'] = idx
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
                "content": f"You are an economic data assistant with access to FRED API. {INDICATOR_GUIDE}"
            },
            {
                "role": "user",
                "content": question
            },
            extraction['raw_response']['message']
        ]
        
        # add tool responses
        for result in api_results:
            if result['success']:
                tool_result = {
                    'series_id': result.get('series_id', ''),
                    'indicator': result.get('indicator_name', ''),
                    'data_points': len(result.get('data', [])),
                    'analysis': result.get('analysis', {})
                }
            else:
                tool_result = {
                    'error': result.get('error', 'Unknown error')
                }
            
            messages.append({
                "role": "tool",
                "content": json.dumps(tool_result, ensure_ascii=False)
            })
        
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
    agent = FredLLMAgent(verbose=verbose)
    return agent.process_question(question)


if __name__ == "__main__":
    questions = [
        "What's the most recent federal funds rate?",
        "What's the current unemployment rate?",
        "Show me the latest GDP data",
        # "How's the real estate market in the past 2 years?",
    ]
    
    agent = FredLLMAgent(verbose=True)
    
    for question in questions:
        result = agent.process_question(question)