from datasets import Dataset
import random
from numpy import indices
import torch
import torch.nn.functional as F

from src.baseline_test import embedding
from src.metrics import cosine_similarity

### Sentence Pairs Strategy
def create_sentence_pairs(dataset, query_embeddings, candidates_embeddings, seed=42):
    random.seed(seed)
    paired_dataset = [] # this will be our final dataset in a List object
    pair = {} # we define a Dict structure for each pair
    ds_length = len(dataset)

    # Iterate over rows of the dataset
    for row_idx, ds_line in enumerate(dataset):
        query = ds_line["query"]
        candidates = ds_line["candidate_chunks"]
        answer_pos = ds_line["answer_pos"]

        positive = candidates[answer_pos] # The correct answer is the positive example for the query

        # Positive pair binding
        pair = { 
            "sentence1": query,
            "sentence2": positive,
            "label": 1.0,
        }
        paired_dataset.append(pair)

        # Remove the current line index to avoid picking the same query as a negative example
        line_idx = list(range(ds_length))
        line_idx.remove(row_idx)

        # Pick 2 random queries from the dataset to create negative pairs with the current query
        for i in range(2):
            random_idx = random.choice(line_idx)
            random_query = dataset[random_idx]["query"]

            # Negative pair binding                
            pair = { 
                "sentence1": query,
                "sentence2": random_query,
                "label": 0.0,
            }
            paired_dataset.append(pair)

            line_idx.remove(random_idx)        
        
        # Compute cosine similarities and sort the negatives by similarity to the query
        similarities = F.cosine_similarity(candidates_embeddings[row_idx], query_embeddings[row_idx].unsqueeze(0)).squeeze()
        sorted_indices = torch.argsort(similarities, descending=True)
        negatives = [candidates[idx] for idx in sorted_indices if idx != answer_pos] # Exclude the positive example from the negatives

        # Pick the top 3 most similar negative candidates and create pairs
        for negative in negatives[:3]:
            # Negative pair binding
            pair = {
                "sentence1": query,
                "sentence2": negative,
                "label": 0.0,
            }
            paired_dataset.append(pair)
        
    return Dataset.from_list(paired_dataset)

### Multiple Negatives Ranking Strategy
def create_batches(dataset, query_embeddings, candidates_embeddings): # Creates batches with multiple negatives
  batch = {
      "anchor": [],
      "positive": [],
      "negative_1": [],
      "negative_2": [],
      "negative_3": []
  }

  # Pick 3 negative candidates at random for each query (every query has at least 5 candidates, answer included)
  for row_idx, ds_line in dataset.iterrows():
    query = ds_line["query"]
    candidates = ds_line["candidate_chunks"]
    n_candidates = ds_line["n_candidates"]
    answer = ds_line["answer"]
    answer_pos = ds_line["answer_pos"]

    # Add the query and the positive example to the batch
    batch["anchor"].append(query)
    batch["positive"].append(answer)

    # Remove the positive example from the candidates to avoid picking it as a negative example
    indices = list(range(n_candidates))
    indices.remove(answer_pos)

    # Compute cosine similarities and sort the negatives by similarity to the query
    similarities = F.cosine_similarity(candidates_embeddings[row_idx], query_embeddings[row_idx].unsqueeze(0)).squeeze()
    sorted_indices = torch.argsort(similarities, descending=True)
    sorted_indices = [idx.item() for idx in sorted_indices if idx.item() != answer_pos]

    # Pick the top 3 most similar negative candidates and add them to the batch
    for i in range(3):
      batch[f"negative_{i+1}"].append(candidates[sorted_indices[i]])

  return Dataset.from_dict(batch)

### Multiple Negatives Ranking Strategy - Evaluation Only
def create_samples(mnr_dev_dataset): # Creates samples to be used for evaluation with the RerankingEvaluator
    samples = []

    # Create a sample for each query in the dev set
    for ds_line in mnr_dev_dataset:
        negatives = []
        for i in range(1, 4):
            negatives.append(ds_line[f"negative_{i}"])
            samples.append({
                "query": ds_line["anchor"],
                "positive": [ds_line["positive"]],
                "negative": negatives
            })
    return samples