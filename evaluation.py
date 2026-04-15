from metrics import compute_metrics, hit_at_k, euclidean_distance, cosine_similarity

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

def generate_jsonl(split_name, exports):
  group_name = "It's_always_loss"
  if split_name == "blind":
    query_ids = query_ids_blind
    query_embeddings = queries_blind_embedding
    candidate_embeddings = candidates_blind_embedding
  else:
    query_ids = query_ids_test
    query_embeddings = query_test
    candidate_embeddings = candidate_chunks_test

  for variant, query_embeddings, candidate_embeddings, metric in exports:
      filename = f"{group_name}-{split_name}-{variant}.jsonl"
      print(f"{filename} generation...")
      create_jsonl(query_ids, query_embeddings, candidate_embeddings, filename, metric)