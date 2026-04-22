from src.metrics import hit_at_k, euclidean_distance, cosine_similarity
import glob
import json
import torch
import gc
import os
from sentence_transformers import SentenceTransformer
from src.data_loader import load_data
from src.baseline_test import embedding

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
    metrics = hit_at_k(dev_query_embeddings, dev_candidates_embeddings, answer_pos, similarity)

    print(f"Hit@K metrics results: {metrics}")

    exports = [(model_name, dev_query_embeddings, dev_candidates_embeddings, similarity)]

    print("JSONL Generation...")
    generate_jsonl("dev", exports, dataset["query_id"], dev_query_embeddings, dev_candidates_embeddings)

    return model_name, metrics

def create_jsonl(query_ids, query_embeddings, candidate_embeddings, filename, similarity):
    with open(filename, 'w') as f:
        for i, q_id in enumerate(query_ids):
          query_embedding = query_embeddings[i].unsqueeze(0)
          candidate_embedding = candidate_embeddings[i]

          if similarity == "cosine":
            similarities = cosine_similarity(query_embedding, candidate_embedding)
            similarities = torch.round(similarities * 1e5) / 1e5
            ranking = torch.argsort(similarities, descending=True, stable=True)
          else:
            if similarity == "euclidean":
              distances = euclidean_distance(query_embedding, candidate_embedding)
              distances = torch.round(distances * 1e5) / 1e5
              ranking = torch.argsort(distances, descending=False, stable=True)

          line = {q_id: ranking.tolist()}
          f.write(json.dumps(line) + '\n')

def generate_jsonl(split_name, exports, query_ids, query_embeddings, candidate_embeddings):
  group_name = "Its_always_loss"

  output_dir = os.path.join("predictions", split_name)

  os.makedirs(output_dir, exist_ok=True)
  
  for variant, query_embeddings, candidate_embeddings, metric in exports:
      filename = f"{group_name}-{split_name}-{variant}-{metric}.jsonl"
      filepath = os.path.join(output_dir, filename)
      print(f"{filename} generation...")
      create_jsonl(query_ids, query_embeddings, candidate_embeddings, filepath, metric)

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
        name, metrics = evaluate_model(run_path, ds["dev"], device, "cosine")
        cosine_similarity_results.append((name, metrics))
        name, metrics = evaluate_model(run_path, ds["dev"], device, "euclidean")
        euclidean_distance_results.append((name, metrics))

    print("\nAll Models Evaluation Results:")
    
    output_file = "dev_evaluation_results.txt"
    output_path = os.path.join("predictions", output_file)
    
    with open(output_path, 'w') as f:
        print(f"Model Evaluation Results (Cosine Similarity):\n")
        f.write("Model Evaluation Results (Cosine Similarity):\n")
        for name, metrics in cosine_similarity_results:
            formatted_metrics = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Model: {name}")
            print(f"Metrics: {formatted_metrics}\n")
            f.write(f"Model: {name}\n")
            f.write(f"Metrics: {formatted_metrics}\n")
        print(f"Model Evaluation Results (Euclidean Distance):\n")
        f.write("\nModel Evaluation Results (Euclidean Distance):\n")
        for name, metrics in euclidean_distance_results:
            formatted_metrics = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Model: {name}")
            print(f"Metrics: {formatted_metrics}\n")
            f.write(f"Model: {name}\n")
            f.write(f"Metrics: {formatted_metrics}\n")
    print("Finish !")

if __name__ == "__main__":
  main()
