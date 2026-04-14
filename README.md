# An Agentic RAG System Based on Macroeconomic Data

This is an agentic Retrieval-Augmented Generation (RAG) system that answers natural language questions about US macroeconomic indicators by autonomously querying the FRED API. Given a user question, the system identifies the relevant FRED data series and date range, retrieves the time-series data via tool calls, and generates a grounded natural language summary—without requiring the user to know any series IDs or API details.

The project systematically compares two models (LLaMA 3.2 and GPT-4o-mini) across three progressively enhanced agent variants: 
1. **Baseline**: Uses a compact text indicator guide for LLM-driven series selection
2. **Semantic Retrieval**: Dynamically selects the most relevant series via FAISS embeddings
3. **Enhanced Version**: Incorporates a rule-based date parser and a three-layer self-check pipeline (relevance gating, parameter validation, and answer completeness verification)

LLaMA 3.2 is further fine-tuned on GPT-generated reference summaries to close the summarization quality gap. Retrieval accuracy is evaluated across all six variants on a curated question benchmark covering single- and multi-series queries, relative time expressions, and out-of-scope questions.

Two directions are planned for future development:
- **News Integration**: Expand the knowledge base with real-time news retrieval via NewsAPI, allowing the system to contextualize economic data with current events and analyst commentary
- **Query Router**: Introduce dynamic strategy selection (direct generation, single-step tool call, or full agentic RAG) based on question type to reduce unnecessary API calls and latency

**Report**: [View PDF](report.pdf)

---

## Table of Contents
- [Get Started](#get-started)
- [Baseline Models with Compact Text Guide](#baseline-models-with-compact-text-indicator-guide)
- [Semantic Retrieval](#semantic-retrieval-instead-of-compact-text)
- [RAG with Self-Check](#rag-with-self-check-and-fallback)
- [Evaluation](#evaluation)
  - [Retrieval Accuracy](#retrieval-accuracy-evaluation)
  - [Summary Quality](#summary-quality-evaluation)
- [Improvements](#improvements)
  - [Few-Shot Prompting](#few-shot-prompt-for-better-summary)
  - [Fine-Tuning](#fine-tune-for-better-summary)
- [User Interface](#user-interface)
- [Future Work](#future-work)

---

## 🚀 Get Started

### Prerequisites
- Python 3.12+
- FRED API Key ([Get one here](https://fred.stlouisfed.org/docs/api/api_key.html))
- Optional: OpenAI API Key for GPT models

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Configure API keys**
   
   Create `src/fred_key.py`:
   ```python
   fred_key = "your_fred_api_key_here"
   ```
   
   Create `src/gpt_key.py`:
   ```python
   gpt_key = "your_openai_api_key_here"
   ```

3. **Generate indicator metadata**
   ```bash
   # Crawl FRED series from Wikipedia tables
   python preparation/wikitable_crawler.py
   
   # Generate compact indicator guide for LLMs
   python preparation/indicator_formatter.py
   ```

---

## Baseline Models with Compact Text Indicator Guide

**Approach**: LLM-driven autonomous understanding using a full text list of indicators.

### Setup

For **LLaMA 3.2**:
```bash
# Install LLaMA 3.2 via Ollama
ollama pull llama3.2
```

For **GPT-4o-mini**:
- Ensure OpenAI API key is configured in `src/gpt_key.py`

### Run Experiments

```bash
# Test with LLaMA 3.2
python src/llama_api.py

# Test with GPT-4o-mini
python src/gpt_api.py
```

**Test Data**: 
- [`data/QA.json`](data/QA.json) - Development set
- [`data/QA_test.json`](data/QA_test.json) - Test set

---

## Semantic Retrieval Instead of Compact Text

**Approach**: Use FAISS embeddings to dynamically retrieve the most relevant series instead of providing the full list.

### Setup

1. **Generate series descriptions** (using GPT-4o-mini)
   ```bash
   python preparation/generate_series_description.py
   ```
   This creates detailed descriptions for each series in `src/output.json`.

2. **Build FAISS index**
   ```bash
   python preparation/build_series_index.py
   ```
   This generates embeddings and builds the FAISS index for retrieval.

### Run Experiments

```bash
# LLaMA 3.2 with semantic retrieval
python src/llama_api_semantic_retriever.py

# GPT-4o-mini with semantic retrieval
python src/gpt_api_semantic_retriever.py
```

---

## RAG with Self-Check and Fallback

<p align="center">
  <img src="figs/self_check_flow_chart.png" alt="Self-check workflow" width="500">
</p>

**Enhancements**:
- **Date Parser**: Rule-based date normalization (e.g., "past 5 years" → actual dates)
- **Three-Layer Self-Check Pipeline**:
  1. Relevance Gating: Verify series selection is appropriate
  2. Parameter Validation: Check date ranges and series IDs
  3. Answer Completeness: Ensure all aspects of the question are addressed

### Run Enhanced Versions

```bash
# LLaMA 3.2 with full enhancements
python src/llama_api_final.py

# GPT-4o-mini with self-checks
python src/gpt_api_final.py
```

---

## Evaluation

### Retrieval Accuracy Evaluation

Evaluates series selection and date range accuracy across all agent variants.

**Run Benchmark**:
```bash
jupyter notebook retrieval_accuracy_benchmark.ipynb
```

**Evaluated Variants**:

| # | File | Model | Components |
|---|------|-------|-----------|
| 1 | [`src/llama_api.py`](src/llama_api.py) | LLaMA 3.2 | Full guide |
| 2 | [`src/llama_api_semantic_retriever.py`](src/llama_api_semantic_retriever.py) | LLaMA 3.2 | Semantic retrieval |
| 3 | [`src/llama_api_final.py`](src/llama_api_final.py) | LLaMA 3.2 | Semantic + date parser + checks |
| 4 | [`src/gpt_api.py`](src/gpt_api.py) | GPT-4o-mini | Full guide |
| 5 | [`src/gpt_api_semantic_retriever.py`](src/gpt_api_semantic_retriever.py) | GPT-4o-mini | Semantic retrieval |
| 6 | [`src/gpt_api_final.py`](src/gpt_api_final.py) | GPT-4o-mini | Semantic + checks |

**Metrics**:
- Series ID F1 Score
- Date Range Accuracy
- Overall Retrieval Success Rate

---

### Summary Quality Evaluation

Evaluates the quality of generated natural language summaries.

**Run Benchmark**:
```bash
jupyter notebook summary_evaluation_benchmark.ipynb
```

**Metrics**:
1. **BERTScore**: Semantic similarity to human-written references ([`files/human_generated_summary_test.json`](files/human_generated_summary_test.json))
2. **Key Fact Coverage Rate**: Percentage of critical facts included
3. **Hallucination Rate**: Detection of fabricated information

**Evaluated Variants**:

**Base Models** (9 variants):

| # | Model | Components |
|---|-------|-----------|
| 1 | GPT-4o-mini | Full guide |
| 2 | GPT-4o-mini | Semantic |
| 3 | GPT-4o-mini | Semantic + checks |
| 4 | LLaMA 3.2 | Full guide |
| 5 | LLaMA 3.2 | Semantic |
| 6 | LLaMA 3.2 | Semantic + date parser + checks |
| 7 | LLaMA 3.2 (fine-tuned) | Full guide |
| 8 | LLaMA 3.2 (fine-tuned) | Semantic |
| 9 | LLaMA 3.2 (fine-tuned) | Semantic + date parser + checks |

**Few-Shot Variants** (3 variants):

| # | Model | Components |
|---|-------|-----------|
| 1 | LLaMA 3.2 (few-shot) | Full guide |
| 2 | LLaMA 3.2 (few-shot) | Semantic |
| 3 | LLaMA 3.2 (few-shot) | Semantic + date parser + checks |

---

## Improvements

### Few-Shot Prompt for Better Summary

Add high-quality example summaries to guide the model.

**Setup**:
1. Add human-written examples to [`src/few_shot_examples.py`](src/few_shot_examples.py)
2. Set `few_shot=True` when running any main file

**Example**:
```python
# In src/llama_api.py
result = agent.process_question(question, few_shot=True)
```

---

### Fine-Tune for Better Summary

Fine-tune LLaMA 3.2 on GPT-generated high-quality summaries.

**Steps**:

1. **Generate training data**
   ```bash
   jupyter notebook preparation/QA_gpt_transformer.ipynb
   ```
   This uses GPT-4o-mini to generate reference summaries.

2. **Fine-tune LLaMA 3.2**
   ```bash
   jupyter notebook fine_tune/llama_finetune.ipynb
   ```
   Choose to download:
   - Full fine-tuned model, or
   - LoRA adapters (more efficient)

3. **Deploy fine-tuned model**
   ```bash
   # Load with Ollama
   ollama create llama3.2-finetuned -f Modelfile
   ```

4. **Use in experiments**
   ```python
   # Modify src/llama_api.py to use fine-tuned model
   agent = FredLLMAgent(model='llama3.2-finetuned')
   ```

---

## User Interface

Interactive Streamlit web interface for the RAG system.

**Launch**:
```bash
streamlit run user_interface.py
```

**Configuration**:
Edit [`user_interface.py`](user_interface.py) to change the underlying model:
```python
# Line ~XX
MODEL = 'llama3.2'  # or 'gpt-4o-mini' or 'llama3.2-finetuned'
```

---

## Future Work

### Planned Enhancements

1. **News Integration**
   - Retrieve real-time news articles from [NewsAPI](https://newsapi.org/)
   - Contextualize economic data with current events
   - Add analyst commentary and market reactions

2. **Intelligent Query Router**
   - Dynamically select retrieval strategy based on question complexity:
     - **Direct Generation**: Simple factual questions
     - **Single-Step Retrieval**: Single indicator queries
     - **Multi-Step RAG**: Complex multi-indicator analysis
   - Reduce API calls and latency for simpler queries

3. **Additional Data Sources**
   - World Bank API
   - Bureau of Labor Statistics (BLS)
   - International Monetary Fund (IMF)
