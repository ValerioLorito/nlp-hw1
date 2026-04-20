from src.metrics import hit_at_k, euclidean_distance, cosine_similarity
import glob
import json
import torch
import os
from sentence_transformers import SentenceTransformer
from src.data_loader import load_data
from src.baseline_test import embedding

def evaluate_model(model_path, dataset, device):
    model_name = os.path.basename(model_path)
    model = SentenceTransformer(model_path)
    model.to(device)

    dev_query_embeddings, dev_cand_embeddings = embedding(dataset["query"], dataset["candidate_chunks"], model.tokenizer, model)
    answer_pos = dataset["answer_pos"] # for dev set
    metrics = hit_at_k(dev_query_embeddings, dev_cand_embeddings, answer_pos, "cosine")

    print(f"Hit@K metrics results: {metrics}")

    exports = [(model_name, dev_query_embeddings, dev_cand_embeddings, "cosine")]

    print("JSONL Generation...")
    generate_jsonl("dev", exports, dataset["dev"]["query_id"], dev_query_embeddings, dev_cand_embeddings)

    return model_name, metrics

def create_jsonl(query_ids, query_embeddings, candidate_embeddings, filename, similarity):
    with open(filename, 'w') as f:
        for i, q_id in enumerate(query_ids):
          query_embedding = query_embeddings[i].unsqueeze(0)
          candidate_embedding = candidate_embeddings[i]

          if similarity == "cosine":
            similarities = cosine_similarity(query_embedding, candidate_embedding)
            ranking = torch.argsort(similarities, descending=True)
          else:
            if similarity == "euclidean":
              distances = euclidean_distance(query_embedding, candidate_embedding)
              ranking = torch.argsort(distances, descending=False)

          line = {q_id: ranking.tolist()}
          f.write(json.dumps(line) + '\n')

def generate_jsonl(split_name, exports, query_ids, query_embeddings, candidate_embeddings):
  group_name = "It's_always_loss"
  
  for variant, query_embeddings, candidate_embeddings, metric in exports:
      filename = f"{group_name}-{split_name}-{variant}.jsonl"
      print(f"{filename} generation...")
      create_jsonl(query_ids, query_embeddings, candidate_embeddings, filename, metric)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    ds = load_data()

    MODELS_DIR = "models" # Directory where the trained models are saved

    pattern = os.path.join(MODELS_DIR, "*", "*", "*")
    all_runs = glob.glob(pattern)

    all_runs = [f for f in all_runs if os.path.isdir(f)]


    all_results = []
    for run_path in all_runs:
        name, metrics = evaluate_model(run_path, ds["dev"], device)
        all_results.append((name, metrics))
    
    print("\nAll Models Evaluation Results:")
    
    for name, metrics in all_results:
        formatted_metrics = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Model: {name}")
        print(f"Metrics: {formatted_metrics}\n")

    print("Finish !")

if __name__ == "__main__":
  main()
