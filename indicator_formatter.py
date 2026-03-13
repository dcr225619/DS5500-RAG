import json
from collections import defaultdict

"""
Format the FRED indicator list into prompts in different ways for LLM
"""

class IndicatorFormatter:
    def __init__(self, json_file='output.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.indicators = json.load(f)
    
    def format_compact_list(self):
        """
        compact format
        """
        lines = ["Available FRED Economic Indicators:\n",
                "Format: - SERIES_ID - Indicator Name - Frequency"
                ]
        
        for ind in self.indicators:
            lines.append(f"- {ind['SERIES']} - {ind['INDICATOR']} - ({ind['PERIOD']})")
        
        return "\n".join(lines)

def generate_all_formats(input_file='D:/NU/DS5500/RAG project/output.json'):
    """generate all formats and save"""
    formatter = IndicatorFormatter(input_file)
    
    formats = {
        'compact': formatter.format_compact_list()
    }
    
    for name, content in formats.items():
        output_file = f'files/indicator_guide_{name}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # compute tokens
        token_count = len(content.split()) * 1.3  # 1.3 token/english word in average 
        print(f"Done: {name:15} format saved to {output_file:30} (~{int(token_count)} tokens)")
    
    return formats

if __name__ == "__main__":
    formats = generate_all_formats()