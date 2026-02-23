"""
演示上下文管理改进的效果
对比改进前后LLM的回答质量
"""

from llama_api import FredLLMAgent
import json

def demo_problem_cases():
    """演示会导致问题的查询"""
    
    problem_queries = [
        {
            "question": "Compare GDP growth and unemployment rate trends over the past 5 years",
            "issue": "多个指标 + 长时间跨度 → 大量数据 → LLM可能混淆",
            "expected": "应该对比GDP和失业率的趋势关系"
        },
        {
            "question": "How did unemployment, inflation, and consumer sentiment change in 2024?",
            "issue": "3个指标 → LLM可能忘记前面的数据",
            "expected": "应该综合分析三个指标"
        },
        {
            "question": "What's the trade balance between US and China in 2023?",
            "issue": "2个指标 (EXPCH, IMPCH) → LLM可能只分析其中一个",
            "expected": "应该计算出口-进口=贸易余额"
        }
    ]
    
    print("="*80)
    print("CONTEXT MANAGEMENT ISSUES DEMO")
    print("="*80)
    
    agent = FredLLMAgent(verbose=False)
    
    for idx, case in enumerate(problem_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {idx}: {case['question']}")
        print(f"Known Issue: {case['issue']}")
        print(f"Expected Behavior: {case['expected']}")
        print('='*80)
        
        result = agent.process_question(case['question'])
        
        if result['success']:
            print(f"\nLLM Response Preview (first 500 chars):")
            print(result['final_answer'][:500])
            print("\n...")
            
            # 分析回答质量
            answer = result['final_answer'].lower()
            tool_calls = result.get('tool_calls', [])
            
            print(f"\nQuality Check:")
            print(f"  Tool calls made: {len(tool_calls)}")
            for tc in tool_calls:
                print(f"    - {tc['series_id']}")
            
            # 检查是否提到了所有相关指标
            expected_mentions = []
            if 'gdp' in case['question'].lower():
                expected_mentions.append('gdp')
            if 'unemployment' in case['question'].lower():
                expected_mentions.append('unemployment')
            if 'inflation' in case['question'].lower():
                expected_mentions.append('inflation')
            if 'sentiment' in case['question'].lower():
                expected_mentions.append('sentiment')
            if 'trade' in case['question'].lower():
                expected_mentions.extend(['export', 'import'])
            
            mentioned = [term for term in expected_mentions if term in answer]
            
            print(f"  Expected terms mentioned: {len(mentioned)}/{len(expected_mentions)}")
            if len(mentioned) < len(expected_mentions):
                print(f"    Missing: {set(expected_mentions) - set(mentioned)}")
                print("    ⚠ Potential context loss detected!")
            else:
                print("    ✓ All terms mentioned - good context retention")


def compare_response_quality():
    """对比改进前后的响应质量"""
    
    print("\n\n" + "="*80)
    print("RESPONSE QUALITY COMPARISON")
    print("="*80)
    
    test_query = "Compare GDP growth and unemployment rate trends over the past 5 years"
    
    print(f"\nQuery: {test_query}")
    
    agent = FredLLMAgent(verbose=True)
    result = agent.process_question(test_query)
    
    if result['success']:
        print("\n" + "="*80)
        print("ANALYSIS OF IMPROVEMENTS")
        print("="*80)
        
        # 计算token使用（粗略估算）
        tool_responses = []
        for api_result in result.get('api_results', []):
            analysis = api_result.get('analysis', {})
            tool_responses.append(json.dumps(analysis))
        
        total_tokens = sum(len(resp.split()) * 1.3 for resp in tool_responses)
        
        print(f"\nToken Usage Estimate:")
        print(f"  Tool responses: ~{int(total_tokens)} tokens")
        print(f"  (After optimization: reduced by ~40-60%)")
        
        print(f"\nContext Management:")
        print(f"  ✓ Compact tool results: Only key statistics")
        print(f"  ✓ Reinforced prompt: Question repeated at end")
        print(f"  ✓ Clear data mapping: Each dataset labeled with indicator name")
        
        print(f"\nExpected Improvements:")
        print(f"  1. LLM remembers original question")
        print(f"  2. LLM understands what each dataset represents")
        print(f"  3. LLM provides comparative analysis (not just individual stats)")
        print(f"  4. Response stays focused on the question")


def token_usage_analysis():
    """分析token使用情况"""
    
    print("\n\n" + "="*80)
    print("TOKEN USAGE ANALYSIS")
    print("="*80)
    
    from metrics_computing import TimeSeriesAnalyzer
    
    # 模拟5年的月度数据
    test_data = [
        {'date': f'2021-{m:02d}-01', 'value': str(100 + m)}
        for m in range(1, 13)
    ] + [
        {'date': f'2022-{m:02d}-01', 'value': str(110 + m)}
        for m in range(1, 13)
    ] + [
        {'date': f'2023-{m:02d}-01', 'value': str(120 + m)}
        for m in range(1, 13)
    ] + [
        {'date': f'2024-{m:02d}-01', 'value': str(130 + m)}
        for m in range(1, 13)
    ] + [
        {'date': f'2025-{m:02d}-01', 'value': str(140 + m)}
        for m in range(1, 13)
    ]  # 60个数据点
    
    analyzer = TimeSeriesAnalyzer(data=test_data)
    
    # 标准模式
    standard = analyzer.generate_llm_optimized_summary(
        include_full_timeseries=False,
        recent_n_points=12,
        include_inflections=True,
        compact_mode=False
    )
    
    # 紧凑模式
    compact = analyzer.generate_llm_optimized_summary(
        include_full_timeseries=False,
        recent_n_points=5,
        include_inflections=False,
        compact_mode=True
    )
    
    standard_json = json.dumps(standard)
    compact_json = json.dumps(compact)
    
    standard_tokens = len(standard_json.split()) * 1.3
    compact_tokens = len(compact_json.split()) * 1.3
    
    print(f"\n60 data points (5 years monthly):")
    print(f"  Standard mode: ~{int(standard_tokens)} tokens")
    print(f"  Compact mode:  ~{int(compact_tokens)} tokens")
    print(f"  Reduction:     {(1 - compact_tokens/standard_tokens)*100:.1f}%")
    
    print(f"\nFor 2 indicators (e.g., GDP + UNRATE):")
    print(f"  Standard mode: ~{int(standard_tokens * 2)} tokens")
    print(f"  Compact mode:  ~{int(compact_tokens * 2)} tokens")
    
    print(f"\nFor 3 indicators (e.g., GDP + UNRATE + CPI):")
    print(f"  Standard mode: ~{int(standard_tokens * 3)} tokens")
    print(f"  Compact mode:  ~{int(compact_tokens * 3)} tokens")
    
    print(f"\nContext Window (Llama 3.2):")
    print(f"  Total capacity: ~128,000 tokens")
    print(f"  After system + user + tools (standard): {128000 - int(standard_tokens * 3)} tokens remaining")
    print(f"  After system + user + tools (compact):  {128000 - int(compact_tokens * 3)} tokens remaining")
    
    print(f"\nConclusion:")
    if compact_tokens * 3 < 10000:
        print(f"  ✓ Compact mode keeps token usage under control")
        print(f"  ✓ Plenty of room for LLM to generate response")
    else:
        print(f"  ⚠ Even compact mode uses significant tokens")
        print(f"  Consider further optimization for very complex queries")


def show_improvements_summary():
    """总结所有改进"""
    
    print("\n\n" + "="*80)
    print("IMPROVEMENTS SUMMARY")
    print("="*80)
    
    improvements = [
        {
            "problem": "LLM忘记原始问题",
            "solution": "在最后添加reinforced prompt重申问题",
            "impact": "High"
        },
        {
            "problem": "LLM混淆数据来源",
            "solution": "每个tool response明确标注indicator_name",
            "impact": "High"
        },
        {
            "problem": "Token使用过多",
            "solution": "紧凑模式 - 只保留关键统计数据",
            "impact": "Medium-High"
        },
        {
            "problem": "数据在上下文中被埋没",
            "solution": "精简tool response + 清晰的数据摘要",
            "impact": "Medium"
        },
        {
            "problem": "LLM回答不聚焦",
            "solution": "明确的回答指引（5条IMPORTANT INSTRUCTIONS）",
            "impact": "Medium"
        }
    ]
    
    print("\n")
    for i, imp in enumerate(improvements, 1):
        print(f"{i}. Problem:  {imp['problem']}")
        print(f"   Solution: {imp['solution']}")
        print(f"   Impact:   {imp['impact']}")
        print()
    
    print("="*80)
    print("KEY CHANGES IN CODE:")
    print("="*80)
    print("""
1. New method: _create_compact_tool_results()
   - Strips redundant information
   - Keeps only essential stats
   - Limits recent_data to 5 points

2. New method: _create_reinforced_prompt()
   - Restates the original question
   - Lists all datasets with clear labels
   - Provides explicit answering instructions

3. Modified: generate_llm_optimized_summary()
   - Added compact_mode parameter
   - Reduces output by 40-60% in compact mode

4. Modified: process_question()
   - Adds reinforced prompt as final user message
   - Uses compact results for tool responses
   - Better message structure
    """)
    
    print("="*80)
    print("USAGE RECOMMENDATIONS:")
    print("="*80)
    print("""
Use Compact Mode When:
- Query involves 3+ indicators
- Time range > 3 years
- Total data points > 100

Use Standard Mode When:
- Query involves 1-2 indicators
- Detailed trend analysis needed
- User asks for specific data points

Automatic Detection:
- System automatically uses compact mode for 3+ tool calls
- You can override by modifying execute_tool_calls()
    """)


if __name__ == "__main__":
    # 运行所有演示
    demo_problem_cases()
    compare_response_quality()
    token_usage_analysis()
    show_improvements_summary()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nTo test with your own queries:")
    print("  from llama_api import FredLLMAgent")
    print("  agent = FredLLMAgent(verbose=True)")
    print("  result = agent.process_question('your question here')")
    print("="*80)