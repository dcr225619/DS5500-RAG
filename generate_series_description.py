import json
import time
from openai import OpenAI
from gpt_api import gpt_key

client = OpenAI(api_key=gpt_key)

with open("output.json", "r", encoding="utf-8") as f:
    series_list = json.load(f)

SYSTEM_PROMPT = """You are an expert economist. For each FRED economic indicator, 
write a concise description (3-4 sentences) that includes:
1. What it measures (plain English definition)
2. Its economic significance and what it indicates
3. How it behaves in different economic conditions (rises/falls when...)
4. Related natural language terms users might use when asking about this topic

Keep descriptions factual and concise. The related terms should reflect how 
non-economists would naturally phrase questions about this topic.

Respond ONLY with a JSON object in this exact format:
{
  "description": "your description here"
}"""

def generate_description(series: dict) -> str:
    user_prompt = f"""Generate a description for this FRED indicator:
Series ID: {series['SERIES']}
Indicator Name: {series['INDICATOR']}
Category: {series['CATEGORY']}
Sub-category: {series['SUB-CATEGORY']}
Units: {series['UNITS']}
Frequency: {series['PERIOD']}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=200
    )

    content = response.choices[0].message.content.strip()
    # strip markdown code blocks if present
    content = content.replace("```json", "").replace("```", "").strip()
    result = json.loads(content)
    return result["description"]


def main():
    temp = []
    total = len(series_list)
    
    for idx, series in enumerate(series_list):
        series_id = series["SERIES"]
        
        print(f"[{idx+1}/{total}] Generating description for {series_id}: {series['INDICATOR']}...")
        
        try:
            description = generate_description(series)
            enriched_series = {**series, "description": description}
            temp.append(enriched_series)
            print(f"  Done: {description[:80]}...")
            
            # save progress after each series in case of interruption
            with open("output_with_descriptions.json", "w", encoding="utf-8") as f:
                json.dump(temp, f, ensure_ascii=False, indent=2)
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Failed for {series_id}: {e}")
            time.sleep(2)
            continue

    print(f"\n{len(temp)} descriptions generated.")
    print("Saved to output_with_descriptions.json")


if __name__ == "__main__":
    main()