from src.metrics import hit_at_k, euclidean_distance, cosine_similarity, mrr_at_k
import glob
import json
import torch
import gc
import os
from sentence_transformers import SentenceTransformer
from src.data_loader import load_data
from src.baseline_test import embedding
from jsonl import generate_jsonl

def evaluate_model(model_path, dataset, device, similarity):
    model_name = os.path.basename(model_path)
    model = SentenceTransformer(model_path)
    model.to(device)

    embedding_file = f"{model_name}_embeddings.pt"

    if os.path.exists("embeddings/" + embedding_file):
        saved_embs = torch.load("embeddings/" + embedding_file, weights_only=True)
        dev_query_embeddings = saved_embs["dev_q"]
        dev_candidates_embeddings = saved_embs["dev_c"]
    else:
        dev_query_embeddings, dev_candidates_embeddings = embedding(dataset["query"], dataset["candidate_chunks"], model.tokenizer, model)
        os.makedirs("embeddings", exist_ok=True)
        torch.save({
            "dev_q": dev_query_embeddings,
            "dev_c": dev_candidates_embeddings
        }, "embeddings/" + embedding_file)

        # Memory cleanup after embedding generation
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    answer_pos = dataset["answer_pos"] # for dev set
    hit_at_k_metrics = hit_at_k(dev_query_embeddings, dev_candidates_embeddings, answer_pos, similarity)
    mrr_at_k_metrics = mrr_at_k(dev_query_embeddings, dev_candidates_embeddings, answer_pos, similarity)

    print(f"Hit@K metrics results: {hit_at_k_metrics}")
    print(f"MRR@K metrics results: {mrr_at_k_metrics}")

    exports = [(model_name, dev_query_embeddings, dev_candidates_embeddings, similarity)]

    print("JSONL Generation...")
    generate_jsonl("dev", exports, dataset["query_id"], dev_query_embeddings, dev_candidates_embeddings)

    return model_name, hit_at_k_metrics, mrr_at_k_metrics

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    ds = load_data()

    MODELS_DIR = "models"

    pattern = os.path.join(MODELS_DIR, "*", "*", "*")
    all_runs = glob.glob(pattern)

    all_runs = [f for f in all_runs if os.path.isdir(f)]

    cosine_similarity_results = []
    euclidean_distance_results = []

    for run_path in all_runs:
        name, hit_at_k_metrics, mrr_at_k_metrics = evaluate_model(run_path, ds["dev"], device, "cosine")
        cosine_similarity_results.append((name, hit_at_k_metrics, mrr_at_k_metrics))
        name, hit_at_k_metrics, mrr_at_k_metrics = evaluate_model(run_path, ds["dev"], device, "euclidean")
        euclidean_distance_results.append((name, hit_at_k_metrics, mrr_at_k_metrics))

    print("\nAll Models Evaluation Results:")
    
    output_file = "dev_evaluation_results.txt"
    output_path = os.path.join("predictions", output_file)

    with open(output_path, 'w') as f:
        print(f"Model Evaluation Results (Cosine Similarity):\n")
        f.write("Model Evaluation Results:\n")
        print("Cosine Similarity:\n")
        f.write("======== Cosine Similarity ========\n")
        for name, hit_at_k_metrics, mrr_at_k_metrics in cosine_similarity_results:
            formatted_hit_at_k_metrics = ", ".join([f"{k}: {v:.4f}" for k, v in hit_at_k_metrics.items()])
            formatted_mrr_at_k_metrics = ", ".join([f"{k}: {v}" for k, v in mrr_at_k_metrics.items()])
            print(f"Model: {name}")
            print(f"{formatted_hit_at_k_metrics}")
            print(f"{formatted_mrr_at_k_metrics}\n")
            f.write(f"Model: {name}\n")
            f.write(f"{formatted_hit_at_k_metrics}\n")
            f.write(f"{formatted_mrr_at_k_metrics}\n\n")
        print("Euclidean Distance:\n")
        f.write("======== Euclidean Distance ========\n")
        for name, hit_at_k_metrics, mrr_at_k_metrics in euclidean_distance_results:
            formatted_hit_at_k_metrics = ", ".join([f"{k}: {v:.4f}" for k, v in hit_at_k_metrics.items()])
            formatted_mrr_at_k_metrics = ", ".join([f"{k}: {v}" for k, v in mrr_at_k_metrics.items()])
            print(f"Model: {name}")
            print(f"{formatted_hit_at_k_metrics}")
            print(f"{formatted_mrr_at_k_metrics}\n")
            f.write(f"Model: {name}\n")
            f.write(f"{formatted_hit_at_k_metrics}\n")
            f.write(f"{formatted_mrr_at_k_metrics}\n\n")
    print("Finish !")

if __name__ == "__main__":
  main()
