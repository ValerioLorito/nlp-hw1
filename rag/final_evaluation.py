import json
import numpy as np
import random
import sklearn.metrics

from output_files import export_annotations

def final_evaluation(file):
    overall_a1_scores = []
    overall_a2_scores = []
    overall_judge_scores = []

    with open(file, "r") as f:
        for line in f:
            item = json.loads(line)
            a1_score = int(item.get("annotator_1", 0))
            a2_score = int(item.get("annotator_2", 0))
            judge_score = int(item.get("llm_judge", 0))
            overall_a1_scores.append(a1_score)
            overall_a2_scores.append(a2_score)
            overall_judge_scores.append(judge_score)

    avg_a1_score = np.mean(overall_a1_scores)
    avg_a2_score = np.mean(overall_a2_scores)
    avg_judge_score = np.mean(overall_judge_scores)

    print(f"Average model's score for annotator 1: {avg_a1_score:.4f}")
    print(f"Average model's score for annotator 2: {avg_a2_score:.4f}")
    print(f"Average model's score for LLM-as-a-Judge: {avg_judge_score:.4f}")
    print(f"Cohen's Kappa between annotators: {sklearn.metrics.cohen_kappa_score(overall_a1_scores, overall_a2_scores):.4f}")
    print(f"Cohen's Kappa between annotator 1 and LLM-as-a-Judge: {sklearn.metrics.cohen_kappa_score(overall_a1_scores, overall_judge_scores):.4f}")
    print(f"Cohen's Kappa between annotator 2 and LLM-as-a-Judge: {sklearn.metrics.cohen_kappa_score(overall_a2_scores, overall_judge_scores):.4f}")


def main():
    export_annotations("rag/evaluation/Annotations-flan-t5-large.xlsx", "rag/evaluation/Its_always_loss-evaluation-flan-t5-large-RAG-JUDGE.jsonl")
    final_evaluation("rag/evaluation/Its_always_loss-evaluation-flan-t5-large-RAG-JUDGE.jsonl")

if __name__ == "__main__":
    main()
