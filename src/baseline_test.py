import torch
import pandas as pd
from IPython.display import display
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Mean pooling function
def mean_pooling(model_output, attention_mask):
  token_embeddings = model_output[0]
  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
  sum = torch.sum(token_embeddings * input_mask_expanded, 1)
  counts = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
  return sum / counts

# Optimized encoding function with batching support
def encode_texts(texts, tokenizer, model, batch_size=32):
  all_embeddings = []
  device = torch.device('mps') if torch.backend.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # Ensures to use GPU if it is possible
  model.to(device)

  for i in range(0, len(texts), batch_size): # Break data into small chunks in order to avoid problems of memory
    batch_texts = texts[i:i + batch_size]
    inputs = tokenizer(batch_texts,
                       padding=True, # Used to have sentences with same length
                       truncation=True, # Cut too long sentences
                       return_tensors='pt').to(device)

    with torch.no_grad(): # Disables 'torch' gradients in order to save time (and performance)
      outputs = model(**inputs)
      embeddings = mean_pooling(outputs, inputs['attention_mask'])
      normalized_embeddings = torch.nn.functional.normalize(embeddings , p=2, dim=1).cpu()  # Normalize and move back to CPU to save GPU memory
      all_embeddings.append(normalized_embeddings)

  return torch.cat(all_embeddings, dim=0)

# Optimized embedding function using batching for queries and candidates
def embedding(queries, candidate_chunks_list, tokenizer, model):
  print("Encoding all queries...")
  # Use the model's native encode if it's a SentenceTransformer, otherwise use our helper
  if isinstance(model, SentenceTransformer):
    query_embeddings = model.encode(queries, batch_size=32, convert_to_tensor=True, normalize_embeddings=True).cpu()
  else:
    query_embeddings = encode_texts(queries, tokenizer, model)

  # Flatten the candidates for better GPU performance
  flat_candidates = [chunk for sublist in candidate_chunks_list for chunk in sublist]

  print("Encoding all candidate chunks...")
  if isinstance(model, SentenceTransformer):
    flat_candidate_embeddings = model.encode(flat_candidates, batch_size=128, convert_to_tensor=True, normalize_embeddings=True).cpu()
  else:
    flat_candidate_embeddings = encode_texts(flat_candidates, tokenizer, model, batch_size=128)

  candidate_embeddings = []
  start_idx = 0

  # Regroup the flattened candidates
  print(f"Regrouping candidate chunks for {len(queries)} queries...")
  for chunks in candidate_chunks_list:
    num_chunks = len(chunks)
    group_emb = flat_candidate_embeddings[start_idx : start_idx + num_chunks]
    candidate_embeddings.append(group_emb)
    start_idx += num_chunks

  return query_embeddings, candidate_embeddings

def main():
    from data_loader import load_data
    from metrics import hit_at_k

    # Load pre-trained models and tokenizers
    distilbert = 'distilbert/distilbert-base-uncased'
    distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert)
    distilbert_model = SentenceTransformer(distilbert)

    all_mini = 'sentence-transformers/all-MiniLM-L6-v2'
    all_mini_tokenizer = AutoTokenizer.from_pretrained(all_mini)
    all_mini_model = SentenceTransformer(all_mini)


    ds = load_data() # Load the dataset using our data loader function

    # Generate embeddings for test and blind sets
    distilbert_query_embeddings, distilbert_candidate_embeddings = embedding(ds["test"]["query"], ds["test"]["candidate_chunks"], distilbert_tokenizer, distilbert_model)
    all_mini_query_embeddings, all_mini_candidate_embeddings = embedding(ds["test"]["query"], ds["test"]["candidate_chunks"], all_mini_tokenizer, all_mini_model)

    print(f"Generated {len(distilbert_query_embeddings)} query embeddings.")
    print(f"Generated {len(distilbert_candidate_embeddings)} lists of candidate embeddings.")

    queries_blind_embedding, candidates_blind_embedding = embedding(ds["blind"]["query"], ds["blind"]["candidate_chunks"], distilbert_tokenizer, distilbert_model)

    # Ensure we use the correct answer positions for the test set
    distilbert_results = hit_at_k(distilbert_query_embeddings, distilbert_candidate_embeddings, ds["test"]["answer_pos"], distance_metric='cosine')
    print(f"Hit@K metrics results for the {distilbert}: {distilbert_results}")

    all_mini_results = hit_at_k(all_mini_query_embeddings, all_mini_candidate_embeddings, ds["test"]["answer_pos"], distance_metric='cosine')
    all_mini_results = {k: f"{v:.4f}" for k, v in all_mini_results.items()}
    print(f"Hit@K metrics results for the {all_mini}: {all_mini_results}")

    data = {
        "Model": [distilbert, all_mini],
        "Hit@1 ": [distilbert_results['Hit@1'], all_mini_results['Hit@1']],
        "Hit@3": [distilbert_results['Hit@3'], all_mini_results['Hit@3']],
        "Hit@5": [distilbert_results['Hit@5'], all_mini_results['Hit@5']]
    }
    df_results = pd.DataFrame(data)
    display(df_results.style.format(precision=4).set_caption("Baselines Comparison").hide(axis='index'))

if __name__ == "__main__":
    main()