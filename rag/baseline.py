import sys
import os
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.data_loader import load_data
from rag.utils import load_model

def baseline(model, tokenizer, query, device):
    prompt = f"Question: {query}\nAnswer: "

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    answer = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7, # Lower temperature for more focused responses
        top_p=0.9, # Use nucleus sampling to keep the most probable tokens (not fixed)
        do_sample=True,
        num_beams=1, # No beam search needed, we want a "strict" answer
    )

    generated_answer = tokenizer.decode(answer[0], skip_special_tokens=True)
    print(f"Generated Answer: {generated_answer}\n")

    if "Answer:" in generated_answer:
        extracted_answer = generated_answer.split("Answer:")[-1].strip()
    else:
        extracted_answer = generated_answer

    return extracted_answer

def main():
    ds = load_data()
    model_name = "google/flan-t5-small"  # You can change this to any other model you want to test
    
    model, tokenizer, device = load_model(model_name, "seq2seq")  # Assuming it's a seq2seq model, change if needed
    queries = ds["test"]["query"]

    answers = {}

    for query in queries[:5]: # Limit to the first 5 queries for testing
        answer = baseline(model, tokenizer, query, device)
        answers[query] = f"Model Answer: {answer}; Real Answer: {ds['test']['answer'][queries.index(query)]}" # Store both model and real answers for comparison

    print("Final Answers:")
    for query, answer in answers.items():
        print(f"Query: {query}\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()