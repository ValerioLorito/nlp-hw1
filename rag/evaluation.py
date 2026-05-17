import json
import nltk
from nltk.translate.meteor_score import meteor_score
from utils import load_model
import os
import re
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.data_loader import load_data

def compute_exact_match(prediction, ground_truths):
    for ground_truth in ground_truths:
        if prediction.strip().lower() == ground_truth.strip().lower():
            return 1
    return 0

def compute_sub_EM(prediction, ground_truths):
    for ground_truth in ground_truths:
        if ground_truth.strip().lower() in prediction.strip().lower():
            return 1
    return 0

def compute_meteor(prediction, ground_truths):
    scores = []
    prediction_tokens = nltk.word_tokenize(prediction.lower())
    for ground_truth in ground_truths:
        ground_truth_tokens = nltk.word_tokenize(ground_truth.lower())
        scores.append(meteor_score([ground_truth_tokens], prediction_tokens))
    return max(scores) if scores else 0

def evaluate_all(prediction, ground_truth): # prediction is a string and ground_truth is a list !!!!
    return {
        "EM" : (compute_exact_match(prediction, ground_truth)),
        "subEM" : (compute_sub_EM(prediction, ground_truth)),
        "METEOR" : (compute_meteor(prediction, ground_truth))
    }

def judge_answers(evaluator, tokenizer, device, query, llm_answer, short_answer):
    prompt = ("You are an expert evaluator. Assess if the LLM Answer is correct based on the Short Answer (a Short Answer is a set of concise and correct answers)."
              "You must respond only with a valid JSON object containing two keys: 'reasoning' (a brief explanation) and 'score' (integer 1 or 0):\n"
              "- reasoning: a brief explanaton of why the LLM answer is correct or not based on the short answer.\n"
              "- score: 1 if the LLM answer is correct, 0 otherwise.\n"
              f"Query: {query}\n"f"Query: {query}\n"
              f"LLM Answer: {llm_answer}\n"
              f"Short Answer: {short_answer}\n"
              "Evaluation:")
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    generated_judgement = evaluator.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_judgement = tokenizer.decode(generated_judgement[0], skip_special_tokens=True).strip()
    
    print(f"Generated Judgement: {generated_judgement}")

    match = re.search(r'"score"\s*:\s*([01])', generated_judgement)
    if match:
        score = match.group(1)
    else:
        score = "0"

    return score

def main():
    judge_model, judge_tokenizer, judge_device = load_model("mistralai/Mistral-7B-Instruct-v0.3", "causal")
    
    ds = load_data()
    queries = ds["test"]["query"]

    ANSWERS_DIR = os.path.join(parent_dir, "rag/answers")
    t5_answers = os.path.join(ANSWERS_DIR, "Its_always_loss-test-flan-t5-large-RAG.jsonl")
    llama_answers = os.path.join(ANSWERS_DIR, "Its_always_loss-llama-3.2-1b-instruct-RAG.jsonl")

    judgements = []

    with open(t5_answers, "r") as f:
        t5_answers = [json.loads(line) for line in f]

    for query_index, query in enumerate(queries[:200]): # Limit to the first 200 queries for testing
        item = ds["test"][query_index]
        query = item["query"]
        query_id = item["query_id"]
        short_answer = item["short_answer"]

        judgement = judge_answers(judge_model, judge_tokenizer, judge_device, query, t5_answers[query_index]['generated_answer'], short_answer)
        print(f"Query: {query}\nT5 Answer: {t5_answers[query_index]['generated_answer']}\nShort Answer: {short_answer}\nJudgement: {judgement}\n")

        judgements.append(judgement)

if __name__ == "__main__":
    main()