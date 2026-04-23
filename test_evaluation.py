from src.metrics import hit_at_k, euclidean_distance, cosine_similarity, mrr_at_k
import torch
import gc
import glob
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

    if os.path.exists("embeddings_test/" + embedding_file):
        saved_embs = torch.load("embeddings_test/" + embedding_file, weights_only=True)
        test_query_embeddings = saved_embs["test_q"]
        test_candidates_embeddings = saved_embs["test_c"]
    else:
        test_query_embeddings, test_candidates_embeddings = embedding(dataset["query"], dataset["candidate_chunks"], model.tokenizer, model)
        os.makedirs("embeddings_test", exist_ok=True)
        torch.save({
            "test_q": test_query_embeddings,
            "test_c": test_candidates_embeddings
        }, "embeddings_test/" + embedding_file)

        # Memory cleanup after embedding generation
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    answer_pos = dataset["answer_pos"] # for dev set
    hit_at_k_metrics = hit_at_k(test_query_embeddings, test_candidates_embeddings, answer_pos, similarity)
    mrr_at_k_metrics = mrr_at_k(test_query_embeddings, test_candidates_embeddings, answer_pos, similarity)

    print(f"Hit@K metrics results: {hit_at_k_metrics}")
    print(f"MRR@K metrics results: {mrr_at_k_metrics}")

    exports = [(model_name, test_query_embeddings, test_candidates_embeddings, similarity)]

    print("JSONL Generation...")
    generate_jsonl("test", exports, dataset["query_id"], test_query_embeddings, test_candidates_embeddings)

    return model_name, hit_at_k_metrics, mrr_at_k_metrics

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    ds = load_data()

    MODELS_DIR = "selected_models"

    pattern = os.path.join(MODELS_DIR, "*", "*")
    all_runs = glob.glob(pattern)

    all_runs = [f for f in all_runs if os.path.isdir(f)]

    results = []

    for run_path in all_runs:
        name, hit_at_k_metrics, mrr_at_k_metrics = evaluate_model(run_path, ds["test"], device, "cosine")
        results.append((name, hit_at_k_metrics, mrr_at_k_metrics))

    print("\nAll Models Evaluation Results:")
    
    output_file = "test_evaluation_results.txt"
    output_path = os.path.join("results", output_file)

    with open(output_path, 'w') as f:
        print(f"Model Evaluation Results (Cosine Similarity):\n")
        f.write("Model Evaluation Results:\n")
        print("Cosine Similarity:\n")
        f.write("======== Cosine Similarity ========\n")
        for name, hit_at_k_metrics, mrr_at_k_metrics in results:
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
