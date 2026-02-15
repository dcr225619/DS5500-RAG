import requests
from fred_key import fred_key
from fred_api import load_indicator_metadata, call_fred_api
import requests
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt

BASE_URL = 'https://api.stlouisfed.org/fred/'
OLLAMA_URL = "http://localhost:11434/api/chat"

INDICATOR_GUIDE =f"""
Available FRED Economic Indicators (use series_id to fetch data):

## Key Economic Indicators
- GDP: US Gross Domestic Product (Q)
- UNRATE: Unemployment Rate (M)
- CPIAUCSL: Consumer Price Index (M)
- FEDFUNDS: Federal Funds Rate (M)

## Banking & Finance
- Interest Rates: T10Y2Y (yield spread), DGS10 (10-yr treasury), TB3MS (3-mo T-bill)
- Fed Assets: WALCL (total assets), TREAST (treasuries), WSHOMCB (MBS)
- Bank Credit: TOTBKCR (total credit), CONSUMER (consumer loans), BUSLOANS (business loans)

## Trade & Exchange Rates
- Exports: EXPCH (China), EXPCA (Canada), EXPMX (Mexico)
- Imports: IMPCH (China), IMPCA (Canada), IMPMX (Mexico)
- FX Rates: DEXUSEU (USD/EUR), DEXJPUS (JPY/USD), DEXCHUS (CNY/USD)

## Employment & Income
- Employment: CE16OV (workforce), EMRATIO (employment ratio)
- Earnings: AHEMAN (hourly earnings), PI (personal income)

## Other Key Metrics
- Housing: PERMIT (permits), HSN1F (sales), MORTGAGE30US (30-yr rate)
- Prices: PPIACO (producer prices), OILPRICE (WTI crude)
- Sentiment: UMCSENT (consumer sentiment), VIXCLS (volatility index)

Note: (M)=Monthly, (Q)=Quarterly, (W)=Weekly, (D)=Daily, (Y)=Yearly

If you need other indicators, search the full metadata.

When asked about economic data, use the get_fred_data function with the appropriate series_id.

Based on the requirements of the task, select the start and end times for the data to be extracted. 

To compare value changes over time, in addition to the data for the year in question, data from five year prior can be retrieved for comparison purposes.

IMPORTANT: 
1. Always specify start_date and end_date based on the user's question. Always use YYYY-MM-DD format, NOT relative dates like "-2y"
2. If no time period specified, use recent 1 year by default
3. If you only need today's or most recent data, use recent 1 year as the time period, so that you have data from a date closest to today as reference
4. Today is {datetime.today().strftime('%Y-%m-%d')}
"""

INDICATORS_MAP = load_indicator_metadata()

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
                    "description": "Start date in YYYY-MM-DD format (e.g., '2026-02-02'). Calculate based on user's time period request."
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format (e.g., '2026-02-02'). Usually today's date."
                }
            },
            "required": ["series_id", "start_date", "end_date"]
        }
    }
]

def call_llm(messages):
    """调用LLM"""
    payload = {
        "model": "llama3.2",
        "messages": messages,
        "tools": TOOLS,
        "stream": False
    }
    
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()

def process_question(question):

    print(f'\nQuestion: {question}')

    start = time.time()
    
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

    # let LLM decide what tool to use
    print("Step 1: LLM analyzing question...")
    result = call_llm(messages)
    print(result)  #####

    if "message" not in result:
        print("\nError: No message in LLM response")
        return
    
    assistant_message = result["message"]

    # check if there are tool calls
    if "tool_calls" not in assistant_message:
        print("\nNo tool calls found")
        print(f"Direct response: {assistant_message.get('content', 'No response')}")
        # LLM直接回答
        messages.append(assistant_message)
        # timing
        print()
        print(f"Total time: {time.time() - start:.2f}s")
        print(f"{'='*60}\n")
        
        return

    messages.append(assistant_message)

    print(f"\nStep 2: Executing tool call(s)...\n")
    # 执行所有tool calls
    for idx, tool_call in enumerate(assistant_message["tool_calls"]):

        # parse parameters
        args = tool_call["function"]["arguments"]
        series_id = args.get('series_id', '').strip()
        start_date = args.get('start_date') or args.get('start', '2020-01-01')
        end_date = args.get('end_date') or args.get('end', datetime.today().strftime('%Y-%m-%d'))
        
        if not series_id:
            print(f"Error: No series_id found in tool call {idx}")
            # 即使出错也要添加tool response
            messages.append({
                "role": "tool",
                "content": json.dumps({"error": "No series_id provided"})
            })
            continue
        
        # call FRED API
        api_result = call_fred_api(series_id, start_date, end_date)

        if api_result['success']:
            observations = api_result['data']
            print(f"Retrieved {len(observations)} data points for tool call {idx}")
            
            # # print 5 recent data 
            # print(f"\n    Recent 5 data for {api_result['indicator_name']}:")
            # for obs in observations[-min(5, len(observations)):]:
            #     print(f"      {obs['date']}: {obs['value']}")

            ## plot data
            # def plot_line(observations, title=None):
            #     # sort by date to ensure chronological order from past to present
            #     #sorted_obs = sorted(observations, key=lambda x: x['date'])
            #     dates = [datetime.strptime(obs['date'], "%Y-%m-%d") for obs in observations]
            #     values = [float(obs['value']) for obs in observations]
            #     plt.figure(figsize=(12, 6))
            #     plt.plot(dates, values, color='r')
            #     if title is not None:
            #         plt.title(title)
            #     plt.xlabel('Date', fontsize=10)
            #     plt.ylabel('Value', fontsize=10)
            #     plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
            #     plt.legend()
            #     plt.grid(True, alpha=0.3) # alpha adjusts transparency
            #     plt.tight_layout()
            #     plt.show()

            # plot_line(observations, title=f'{api_result['indicator_name']}')

            tool_result = {
                'series_id': series_id,
                'indicator': api_result['indicator_name'],
                'period': f"{start_date} to {end_date}",
                'raw_data': api_result.get('raw_data', {}),
                'analysis': api_result.get('analysis', {})  # ← 只传分析结果
            }

        else:
            print(f"{api_result['error']}")
            tool_result = api_result
        
        # add tool responds to message
        messages.append({
            "role": "tool",
            "content": json.dumps(tool_result)
        })
                    
    print()
    print("\nStep 3: LLM generating summary based on data...")
    
    final_result = call_llm(messages)
    final_answer = final_result.get("message", {}).get("content", "No response generated")
    
    print(final_answer)
    
    # timing
    print()
    print(f"Total time: {time.time() - start:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # test cases
    questions = [
        #"How's the real estate market in the past 2 years?",
        "What patterns did unemployment and inflation rates show in 2024?",
        #"What's the most recent federal funds rate?",
    ]
    
    for question in questions:
        process_question(question)