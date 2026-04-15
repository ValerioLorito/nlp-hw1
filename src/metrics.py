import torch

def cosine_similarity(query_embedding, chunk_embedding):
  return torch.nn.functional.cosine_similarity(query_embedding, chunk_embedding, dim=1)

def hit_at_k(query_embeddings, candidate_embeddings, answer_pos, distance_metric, ks=[1, 3, 5]):
    hit_counts = {k: 0 for k in ks}
    N = len(query_embeddings)

    for i, query in enumerate(query_embeddings):
      # candidate_embeddings[i] is a tensor of all candidate vectors for query i
      candidates = candidate_embeddings[i]

      if distance_metric == 'cosine':
          scores = torch.nn.functional.cosine_similarity(query.unsqueeze(0), candidates, dim=1)
          descending = True
      elif distance_metric == 'euclidean':
          scores = torch.cdist(query.unsqueeze(0), candidates, p=2).squeeze(0)
          descending = False

      # Get indices of candidates ranked by distance
      sorted_indices = torch.argsort(scores, descending=descending)

      # .item() gets the integer value from the single-element tensor
      rank = (sorted_indices == answer_pos[i]).nonzero(as_tuple=True)[0].item()

      # If rank is 0, it's the top 1. If rank < 3, it's in top 3, etc.
      for k in ks:
          if rank < k:
              hit_counts[k] += 1

    # Average the hits over all queries
    return {f"Hit@{k}": hit_counts[k] / N for k in ks}

def euclidean_distance(query_embedding, chunk_embeddings):
  query_in_box = query_embedding.unsqueeze(0) # adding a dimension to fit with chunk_embeddings dimension for computation
  distances = torch.cdist(query_in_box, chunk_embeddings, p=2) # compute the distance between the query and all chunks
  # p = 2 is for euclidian distance
  return distances.squeeze(0) # remove the dimension and return distances

