import json
import numpy as np
import random
import sklearn.metrics

from output_files import export_annotations

def final_evaluation(file):
    overall_scores = []
    overall_judge_scores = []

    with open(file, "r") as f:
        for line in f:
            item = json.loads(line)
            score = int(item.get("annotator_1", 0))
            judge_score = int(item.get("llm_judge", 0))
            overall_scores.append(score)
            overall_judge_scores.append(judge_score)

    avg_score = np.mean(overall_scores)
    avg_judge_score = np.mean(overall_judge_scores)

    print(f"Average model's score for annotators: {avg_score:.4f}")
    print(f"Average model'sscore for LLM-as-a-Judge: {avg_judge_score:.4f}")
    print(f"Cohen's Kappa between annotators and LLM-as-a-Judge: {sklearn.metrics.cohen_kappa_score(overall_scores, overall_judge_scores):.4f}")

def main():
    export_annotations("rag/evaluation/Annotations-flan-t5-large.xlsx", "rag/evaluation/Its_always_loss-evaluation-flan-t5-large-RAG-JUDGE.jsonl")
    final_evaluation("rag/evaluation/Its_always_loss-evaluation-flan-t5-large-RAG-JUDGE.jsonl")

if __name__ == "__main__":
    main()
