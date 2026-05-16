import nltk
from nltk.translate.meteor_score import meteor_score

from utils import load_model

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

def judge_answers(evaluator, query, llm_answer, short_answer):
    prompt = ("Given an answer to a query, assess if the answer is correct or not with respect to the gold (short) answer. Answer with '1' for Correct or '0' for Incorrect.\n\n"
              f"Query: {query}\n"
              f"LLM Answer: {llm_answer}\n"
              f"Short Answer: {short_answer}\n"
              "Evaluation:")
    
    inputs = prometheus_tokenizer(prompt, return_tensors="pt", truncation=True).to(prometheus_device)
    
    judgement = prometheus_model.generate(**inputs, max_new_tokens=10)

    return judgement

def main():
    prometheus_model, prometheus_tokenizer, prometheus_device = load_model("prometheus-eval/prometheus-7b-v2.0", "causal")
    
    ds = load_data()
    queries = ds["test"]["query"]
    t5_answers = "answers/Its_always_loss-test-flan-t5-large-RAG.jsonl"
    llama_answers = "answers/Its_always_loss-llama-3.2-1b-instruct-RAG.jsonl"

    judgements = []

    for query in queries[:200]: # Limit to the first 200 queries for testing
        item = ds["test"][queries.index(query)]
        query = item["query"]
        short_answer = item["short_answer"]

        judgement = judge_answers(prometheus_model, query, t5_answers[queries.index(query)][generated_answer], short_answer)

        judgements.append(judgement)

if __name__ == "__main__":
    main()