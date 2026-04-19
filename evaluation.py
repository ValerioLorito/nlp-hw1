from src.metrics import hit_at_k, euclidean_distance, cosine_similarity
import json
import torch
import os
from sentence_transformers import SentenceTransformer
from src.data_loader import load_data
from src.baseline_test import embedding

def evaluate_model(model, query_embeddings, candidate_embeddings, answer_pos, distance_metric):
    metrics = hit_at_k(query_embeddings, candidate_embeddings, answer_pos, distance_metric)
    
    print(f"Hit@K metrics results: {metrics}")

    return metrics

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
    model_path = "/home/emie/Documents/Italie/mNLP/HW1/models/distilbert/distilbert-base-uncased-mnr" 
    ds = load_data()

    model = SentenceTransformer(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    test_query_embeddings, test_cand_embeddings = embedding(ds["test"]["query"], ds["test"]["candidate_chunks"], model.tokenizer, model)
    # blind_query_embeddings, blind_cand_embeddings = embedding(ds["blind"]["query"], ds["blind"]["candidate_chunks"], model.tokenizer, model)
    answer_pos = ds["test"]["answer_pos"] # for test set
    results = evaluate_model(model, test_query_embeddings, test_cand_embeddings, answer_pos, 'cosine') # or euclidean
    
    print("Results")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    

    exports = [(os.path.basename(model_path), test_query_embeddings, test_cand_embeddings, "cosine")]

    print("JSONL Generation...")
    generate_jsonl("test", exports, ds["test"]["query_id"], test_query_embeddings, test_cand_embeddings)
    # generate_jsonl("blind", exports, ds["blind"]["query_id"], blind_query_embeddings, blind_cand_embeddings)

    print("Finish !")

if __name__ == "__main__":
  main()

"""
distilbert-base-uncased-pairs :
Hit@K metrics results: {'Hit@1': 0.5855, 'Hit@3': 0.7915, 'Hit@5': 0.862}
Results
Hit@1: 0.5855
Hit@3: 0.7915
Hit@5: 0.8620



"""