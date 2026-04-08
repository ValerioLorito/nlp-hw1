import numpy
import random
import torch
import pandas as pd
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from transformers import (
    #AutoModelForTokenClassification, # I think we want to do token classification but classification on embeddings
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoTokenizer
)

# First, we load the dataset!
ds = load_dataset(
    "sapienzanlp-course-materials/hw-mnlp-2026"
)

# Dataset features initialization
query = ds["train"]["query"]
candidate_chunks = ds["train"]["candidate_chunks"]
n_candidates = ds["train"]["n_candidates"]
answer = ds["train"]["answer"]
answer_pos = ds["train"]["answer_pos"]

query_test = ds["test"]["query"]
candidate_chunks_test = ds["test"]["candidate_chunks"]
n_candidates_test = ds["test"]["n_candidates"]
answer_test = ds["test"]["answer"]
answer_pos_test = ds["test"]["answer_pos"]


# Model initialization
bert_name = 'distilbert/distilbert-base-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
bert_model = AutoModel.from_pretrained(bert_name)

all_mini_name = 'sentence-transformers/all-MiniLM-L6-v2'
all_mini_tokenizer = AutoTokenizer.from_pretrained(all_mini_name)
all_mini_model = SentenceTransformer(all_mini_name)

# mean pooling function
def mean_pooling(model_output, attention_mask):
  token_embeddings = model_output[0]
  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
  sum = torch.sum(token_embeddings * input_mask_expanded, 1)
  counts = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
  return sum / counts

# Optimized encoding function with batching support
def encode_texts(texts, tokenizer, model, batch_size=32):
  all_embeddings = []
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Ensures to use GPU if it is possible
  model.to(device)

  for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad(): # Disables 'torch' gradients in order to save time (and performance)
      outputs = model(**inputs)
      embeddings = mean_pooling(outputs, inputs['attention_mask'])
      # Normalize and move back to CPU to save GPU memory
      normalized_embeddings = torch.nn.functional.normalize(embeddings , p=2, dim=1).cpu()
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

bert_query_embeddings, bert_candidate_embeddings = embedding(query_test, candidate_chunks_test, bert_tokenizer, bert_model)

all_mini_query_embeddings, all_mini_candidate_embeddings = embedding(query_test, candidate_chunks_test, all_mini_tokenizer, all_mini_model)

print(f"Generated {len(bert_query_embeddings)} query embeddings.")
print(f"Generated {len(bert_candidate_embeddings)} lists of candidate embeddings.")

def cosine_similarity(query_embedding, chunk_embedding):
  return torch.nn.functional.cosine_similarity(query_embedding, chunk_embedding, dim=1)

def hit_at_k(query_embeddings, candidate_embeddings, answer_pos, ks=[1, 3, 5]):
  hit_counts = {k: 0 for k in ks}
  N = len(query_embeddings)

  for i, query in enumerate(query_embeddings):
    # Calculate cosine similarities between the query and all its candidates
    # candidate_embeddings[i] is a tensor of all candidate vectors for query i
    candidates = candidate_embeddings[i]
    similarities = cosine_similarity(query, candidates)

    # Get indices of candidates ranked by similarity (highest first)
    sorted_indices = torch.argsort(similarities, descending=True)

    # .item() gets the integer value from the single-element tensor
    rank = (sorted_indices == answer_pos[i]).nonzero(as_tuple=True)[0].item()

    # If rank is 0, it's the top 1. If rank < 3, it's in top 3, etc.
    for k in ks:
      if rank < k:
        hit_counts[k] += 1

  # Average the hits over all queries
  return {f"Hit@{k}": hit_counts[k] / N for k in ks}

# Ensure we use the correct answer positions for the test set
bert_results = hit_at_k(bert_query_embeddings, bert_candidate_embeddings, answer_pos_test)
print(f"Hit@K metrics results for the {bert_name}: {bert_results}")

all_mini_results = hit_at_k(all_mini_query_embeddings, all_mini_candidate_embeddings, answer_pos_test)
all_mini_results = {k: f"{v:.4f}" for k, v in all_mini_results.items()}
print(f"Hit@K metrics results for the {all_mini_name}: {all_mini_results}")

def create_pair(ds_line):
  query = ds_line["query"]
  candidates = ds_line["candidate_chunks"]
  answer_pos = ds_line["answer_pos"]

  positive = candidates[answer_pos] # positive = the good answer
  # for the negative, we list the possible answers and remove the good one.
  indices = list(range(len(candidates)))
  indices.remove(answer_pos)
  # among these, we choose a random one
  negative_i = random.choice(indices)
  negative = candidates[negative_i]

  data_frame = pd.DataFrame({
    "query": [query, query],
    "candidate": [positive, negative],
    "label": [1, 0]
  })
  return data_frame

def sentence_pairs(dataset):
    pairs = []

    for ds_line in dataset:
        pairs.append(create_pair(ds_line))

    return pd.concat(pairs)

train_sentence_pairs = sentence_pairs(ds["train"])

print(f"Sentence pairs for the first query : {train_sentence_pairs.head(2).to_string(index=False)}")

loss = losses.ContrastiveLoss(model=bert_model)

train_args=SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/mpnet-base-all-nli-triplet",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=0.1,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet", 
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=train_sentence_pairs,
)
