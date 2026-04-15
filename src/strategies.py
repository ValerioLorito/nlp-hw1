from datasets import Dataset
import random

### Sentence Pair Strategy
def create_sentence_pairs(dataset):
    paired_dataset = [] # this will be our final dataset in a List object
    pair = {} # we define a Dict structure for each pair

    # Iterate over rows of the dataset
    for _, ds_line in dataset.iterrows():
        query = ds_line["query"]
        candidates = ds_line["candidate_chunks"]
        answer_pos = ds_line["answer_pos"]

        positive = candidates[answer_pos] # positive = the good answer

        pair = { # positive pair binding
            "query": query,
            "candidate": positive,
            "label": 1,
        }
        paired_dataset.append(pair)

        # for the negative, we list the possible answers and remove the good one.
        indices = list(range(len(candidates)))
        indices.remove(answer_pos)
        # among these, we choose a random one
        negative_i = random.choice(indices)
        negative = candidates[negative_i]

        pair = { # negative pair binding
            "query": query,
            "candidate": negative,
            "label": 0,
        }
        paired_dataset.append(pair)

    return Dataset.from_list(paired_dataset) # We use the 'from_list' method in order to retrieve the Dataset in the right structure


### Multiple Negatives Ranking Loss batching strategy
def create_batches(dataset):
  batch = {
      "anchor": [],
      "positive": [],
      "negative_1": [],
      "negative_2": [],
      "negative_3": []
  }

  # We pick k negative candidates at random for each query, where k = 3 (every query has at least 5 candidates, answer included)
  for _, ds_line in dataset.iterrows():
    query = ds_line["query"]
    candidates = ds_line["candidate_chunks"]
    n_candidates = ds_line["n_candidates"]
    answer = ds_line["answer"]
    answer_pos = ds_line["answer_pos"]

    batch["anchor"].append(query)
    batch["positive"].append(answer)

    indices = list(range(n_candidates))
    indices.remove(answer_pos)
    negative_position = random.sample(indices, 3)

    for i, position in enumerate(negative_position):
      batch[f"negative_{i+1}"].append(candidates[position])

  return Dataset.from_dict(batch)

def create_samples(mnr_dev_dataset):
  samples = []

  for ds_line in mnr_dev_dataset:
    negatives = []
    for i in range(1, 4):
      negatives.append(ds_line[f"negative_{i}"])
    samples.append({
        "query": ds_line["anchor"],
        "positive": ds_line["positive"],
        "negative": negatives
    })
  return samples