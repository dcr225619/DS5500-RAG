from llama_api import process_question

questions = [
        #"How's the real estate market in the past 2 years?",
        #"What patterns did unemployment and inflation rates show in 2024?",
        "What's the most recent federal funds rate?",
    ]
    
for question in questions:
    process_question(question)