
import pandas as pd
import json
import requests
from io import StringIO

url = "https://en.wikipedia.org/wiki/Federal_Reserve_Economic_Data"

# add User-Agent to avoid 403 error
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# get the page content
response = requests.get(url, headers=headers)
response.raise_for_status()

# find the tables in the page
tables = pd.read_html(StringIO(response.text))

print(f"Found {len(tables)} tables\n")

# for i, table in enumerate(tables):
#     print(f"Table {i}'s column names: {table.columns.tolist()}")
#     print(f"# of rows : {len(table)}\n")

# the first table is what we need
df = tables[0]

# transform into json
import json
json_data = df.to_json(orient='records', force_ascii=False)

# save to file
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(json.loads(json_data), f, ensure_ascii=False, indent=2)