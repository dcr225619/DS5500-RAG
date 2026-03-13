# An Agentic RAG system based on Macroeconomic Data

## Get Started

1. Install the required dependencies with: `pip install -r requirements.txt --upgrade`.
2. Save your fred api key in a file named `fred_key.py`, save your openai api key in a file named `gpt_key.py`.
3. Run `wikitable_crawler.py` to get series ids file `output.json` for Fred series.
4. Run `indicator_formatter.py` to generate `indicator_guide_compact.txt` for generate a compact Fred data list for llms.

## Use baseline models with compact text indicator guide (llama3.2 / gpt-4o-mini). 
1. Install llama3.2 in Docker if you want to use llama3.2 for experiment. Apply for OpenAI key to use gpt models for experiment.
2. Run `llama_api.py` to use llama3.2 for experiment on `QA.json` or `QA_test.json`. Run `gpt_api.py` to use chatgpt-mini-4o for experiment on `QA.json` or`QA_test.json`. 
3. Run `retrieval_accuracy_test.py` to test retrieval accuracy without generating summary on `QA.json` or `QA_test.json`. Modify parameters to change the model you use.
4. Run `retrieval_result_analysis.ipynb` to analyse the results you get from `retrieval_accuracy_test.py`.

## Use Semantic Retriever insted of compact text indicator guide
1. Run `generate_series_description.py` to generate detailed descriptions for Fred series file `output.json` using chatgpt-mini-4o.
2. Run `build_series_index.py` to build series index embedding for retriever.
3. Run `llama_api_semantic_retriever.py` to use llama3.2 for experiment on `QA.json` with your newly generated semantic retriever.
4. Run `retrieval_accuracy_test.py` to test retrieval accuracy without generating summary on `QA.json`. Modify parameters to change the model you use.

## Fine-tune for better summary
1. Run `QA_gpt_transformer.ipynb` to generate QA results using chatgpt-4o-mini for model fine-tuning on summary generation.
2. Run `llama_finetune.ipynb` for model fine-tuning. Download the correct format of fine-tuned model or LoRA adapters according to your need. 
3. Deploy your fine-tuned model.
4. Modify parameters to run the files using your fine-tuned model.