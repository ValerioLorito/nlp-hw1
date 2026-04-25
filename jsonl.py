import json
import torch
import os
from src.metrics import euclidean_distance, cosine_similarity

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
       
    for variant, query_embeddings, candidate_embeddings, similarity in exports:
        filename = f"{group_name}-{split_name}-{variant}-{similarity}.jsonl"
        filepath = os.path.join(output_dir, filename)
        print(f"{filename} generation...")
        create_jsonl(query_ids, query_embeddings, candidate_embeddings, filepath, similarity)
