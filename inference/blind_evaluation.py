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

    if os.path.exists("embeddings_blind/" + embedding_file):
        saved_embs = torch.load("embeddings_blind/" + embedding_file, weights_only=True)
        blind_query_embeddings = saved_embs["blind_q"]
        blind_candidates_embeddings = saved_embs["blind_c"]
    else:
        blind_query_embeddings, blind_candidates_embeddings = embedding(dataset["query"], dataset["candidate_chunks"], model.tokenizer, model)
        os.makedirs("embeddings_blind", exist_ok=True)
        torch.save({
            "blind_q": blind_query_embeddings,
            "blind_c": blind_candidates_embeddings
        }, "embeddings_blind/" + embedding_file)

        # Memory cleanup after embedding generation
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    exports = [(model_name, blind_query_embeddings, blind_candidates_embeddings, similarity)]

    print("JSONL Generation...")
    generate_jsonl("blind", exports, dataset["query_id"], blind_query_embeddings, blind_candidates_embeddings)

    return model_name

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    ds = load_data()

    MODELS_DIR = "selected_models"

    pattern = os.path.join(MODELS_DIR, "*", "*")
    all_runs = glob.glob(pattern)

    all_runs = [f for f in all_runs if os.path.isdir(f)]

    results = []

    for run_path in all_runs:
        name = evaluate_model(run_path, ds["blind"], device, "cosine")

    print("Finish !")

if __name__ == "__main__":
  main()
