from src.metrics import hit_at_k, euclidean_distance, cosine_similarity
import json
import torch

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
    import torch
    from sentence_transformers import SentenceTransformer
    from src.data_loader import load_data
    from src.baseline_test import embedding

    ds = load_data()

    model_path = "/home/emie/Documents/Italie/mNLP/swisstransfer_986eead0-7514-49ed-92a8-37574b64bb4a/models/distilbert/distilbert-base-uncased-pairs"
    model = SentenceTransformer(model_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    print("Génération des embeddings...")
    test_query_embeddings, test_cand_embeddings = embedding(ds["test"]["query"], ds["test"]["candidate_chunks"], model.tokenizer, model)
    blind_query_embeddings, blind_cand_embeddings = embedding(ds["blind"]["query"], ds["blind"]["candidate_chunks"], model.tokenizer, model)

    exports = [("model_name", test_query_embeddings, test_cand_embeddings, "cosine")]

    print("JSONL Generation...")
    generate_jsonl("test", exports, ds["test"]["query_id"], test_query_embeddings, test_cand_embeddings)
    generate_jsonl("blind", exports, ds["blind"]["query_id"], blind_query_embeddings, blind_cand_embeddings)

    print("Finish !")

if __name__ == "__main__":
  main()