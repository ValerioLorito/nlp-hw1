import json
import nltk
from nltk.translate.meteor_score import meteor_score
import os
import random
import re
import sys
import numpy as np
from tqdm import tqdm
from utils import load_model, set_seed

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.data_loader import load_data
from wikidata_utils import get_wikidata_ground_truth
from output_files import generate_jsonl_file, export_judge_to_excel

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
    prompt = ("You are an expert evaluator. Assess if\te the LLM Answer is correct based on the Short Answer (a Short Answer is a set of concise and correct answers)."
              "You must respond only with a valid JSON object containing two keys: 'reasoning' (a brief explanation) and 'score' (integer 1 or 0):\n"
              "- reasoning: a brief explanation of why the LLM answer is correct or not based on the short answer.\n"
              "- score: 1 if the LLM answer is correct, 0 otherwise.\n"
              f"Query: {query}\n"
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

    match = re.search(r'"score"\s*:\s*([01])', generated_judgement)
    if match:
        score = match.group(1)
    else:
        score = "0"

    return score

def judge_model(model, queries):
    judge_model, judge_tokenizer, judge_device = load_model("mistralai/Mistral-7B-Instruct-v0.3", "causal")

    ANSWERS_DIR = os.path.join(parent_dir, "rag/answers")
    model_answers_path = os.path.join(ANSWERS_DIR, f"Its_always_loss-test-{model}-RAG.jsonl")

    judgements = []
    model_answers = {}

    with open(model_answers_path, "r") as f:
        for line in f:
            data = json.loads(line)
            model_answers[data["query_id"]] = data

    for query, query_id, short_answer in tqdm(queries, desc="Judging Answers"): # Limit to the first 200 queries for testing
        rag_item = model_answers.get(query_id)

        judgement = judge_answers(judge_model, judge_tokenizer, judge_device, query, rag_item['generated_answer'], short_answer)

        # results for JSONL generation
        result_dict = {
            "query_id": rag_item["query_id"],
            "retrieved_chunks": rag_item.get("retrieved_chunks", []),
            "augmented_prompt": rag_item.get("augmented_prompt", ""),
            "generated_answer": rag_item["generated_answer"],
            "llm_judge_output": judgement, 
            "annotator_1": "",
            "annotator_2": ""
        }
        judgements.append(result_dict)

    return judgements

def evaluate_generations(model, query_ids, short_answers, wikidata_answers, system, file): 
    gold_answers = dict(zip(query_ids, short_answers))

    output_file = file.replace(".jsonl", "_scored.jsonl")

    results = []
    total_scores = []
    total_wikidata_scores = []

    with open(file, "r") as f:
        for line in f:
            item = json.loads(line)
            query_id = item["query_id"]
            generated_answer = item["generated_answer"]

            short_answer = gold_answers.get(query_id, "")

            scores = evaluate_all(generated_answer, short_answer)

            wikidata_scores = evaluate_all(generated_answer, wikidata_answers[query_ids.index(query_id)])

            total_scores.append(scores)
            total_wikidata_scores.append(wikidata_scores)

            item["scores"] = scores
            item["wikidata_scores"] = wikidata_scores
            results.append(item)
            

    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    for metric in ["EM", "subEM", "METEOR"]:
        avg_score = sum(score[metric] for score in total_scores) / len(total_scores)
        avg_wikidata_score = sum(score[metric] for score in total_wikidata_scores) / len(total_wikidata_scores)
        print(f"Average {metric} for {model} on {system}: {avg_score:.4f}")
        print(f"Average {metric} for {model} on {system} (using Wikidata answers as ground truth): {avg_wikidata_score:.4f}")

def final_evaluation(file):
    overall_scores = []
    overall_judge_scores = []

    with open(file, "r") as f:
        for line in f:
            item = json.loads(line)
            scores = item.get("annotator_1")
            judge_score = item.get("llm_judge")
            overall_scores.append(scores)
            overall_judge_scores.append(judge_score)
    
    avg_score = sum(overall_scores) / len(overall_scores)
    avg_judge_score = sum(overall_judge_scores) / len(overall_judge_scores)

    print(f"Average score for annotators: {avg_score:.4f}")
    print(f"Average score for LLM judge: {avg_judge_score:.4f}")
    print(f"Correlation between annotators and LLM judge: {np.corrcoef(overall_scores, overall_judge_scores)[0, 1]:.4f}")

def main():
    set_seed()

    ds = load_data()
    queries = ds["test"]["query"]
    query_ids = ds["test"]["query_id"]
    short_answers = ds["test"]["short_answer"]
    wikidata_ids = ds["test"]["wikidata_id"]
    
    wikidata_answers = []

    for query_id, wikidata_id, short_answer in tqdm(zip(query_ids, wikidata_ids, short_answers), total=len(query_ids), desc="Fetching Wikidata Ground Truths"):
        wikidata_answers.append(get_wikidata_ground_truth(wikidata_id, short_answer[0]))

    evaluate_generations("t5", query_ids, short_answers, wikidata_answers, "Baseline", "rag/answers/Its_always_loss-test-flan-t5-large-Baseline.jsonl")
    evaluate_generations("llama", query_ids, short_answers, wikidata_answers, "Baseline", "rag/answers/Its_always_loss-test-llama-3.2-1b-instruct-Baseline.jsonl")

    evaluate_generations("t5", query_ids, short_answers, wikidata_answers, "RAG", "rag/answers/Its_always_loss-test-flan-t5-large-RAG.jsonl")
    evaluate_generations("llama", query_ids, short_answers, wikidata_answers, "RAG", "rag/answers/Its_always_loss-test-llama-3.2-1b-instruct-RAG.jsonl")

    evaluate_generations("t5", query_ids, short_answers, wikidata_answers, "Oracle", "rag/answers/Its_always_loss-test-flan-t5-large-Oracle.jsonl")
    evaluate_generations("llama", query_ids, short_answers, wikidata_answers, "Oracle", "rag/answers/Its_always_loss-test-llama-3.2-1b-instruct-Oracle.jsonl")

    '''selected_queries = random.sample(list(zip(queries, query_ids, short_answers)), 250) # Select a random subset of 250 queries for judging
    queries, query_ids, short_answers = zip(*selected_queries)

    judgements_t5 = judge_model("flan-t5-large", selected_queries)
    generate_jsonl_file(judgements_t5, "evaluation", "flan-t5-large", "RAG-JUDGE", "LLM_judge")
    jsonl_path = f"rag/evaluation/Its_always_loss-evaluation-flan-t5-large-RAG-JUDGE.jsonl"
    export_judge_to_excel(jsonl_path, f"rag/evaluation/Annotations-flan-t5-large.xlsx", queries, short_answers)

    judgements_llama = judge_model("Llama-3.2-1b-instruct", queries, ds)
    generate_jsonl_file(judgements_llama, "evaluation", "Llama-3.2-1b-instruct", "RAG-JUDGE", "LLM_judge")
    jsonl_path = f"rag/evaluation/Its_always_loss-evaluation-Llama-3.2-1b-instruct-RAG-JUDGE.jsonl"
    export_judge_to_excel(jsonl_path, f"rag/evaluation/Annotations-Llama-3.2-1b-instruct.xlsx", queries)'''

if __name__ == "__main__":
    main()