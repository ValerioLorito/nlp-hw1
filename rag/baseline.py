import sys
import os
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.data_loader import load_data
from src.baseline_test import embedding
from rag.utils import load_model

def baseline(model, tokenizer, query, device):
    prompt = f"Question: {query}\nAnswer: "

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    answer = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7, # Lower temperature for more focused responses
        top_p=0.9, # Use nucleus sampling to keep the most probable tokens (not fixed)
        do_sample=True,
        num_beams=1, # No beam search needed, we want a "strict" answer
    )

    generated_answer = tokenizer.decode(answer[0], skip_special_tokens=True)
    print(f"Generated Answer: {generated_answer}\n")

    if "Answer:" in generated_answer:
        extracted_answer = generated_answer.split("Answer:")[-1].strip()
    else:
        extracted_answer = generated_answer

    return extracted_answer


def get_top_k_chunks(query, candidate_chunks, retriever_model, k=3):  
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    chunk_embeddings = retriever_model.encode(candidate_chunks, convert_to_tensor=True)
    
    similarity = util.semantic_search(query_embedding, chunk_embeddings, top_k=k)
    
    top_k_indices = [hit['corpus_id'] for hit in similarity[0]]
    top_k_texts = [candidate_chunks[i] for i in top_k_indices]
    
    return top_k_texts, top_k_indices


def baseline_rag(model, tokenizer, query, retrieved_passages, device):
    context = "\n".join(retrieved_passages) # concatenation of retrieved passages
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    answer = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7, 
        top_p=0.9, 
        do_sample=True,
        num_beams=1,
    )

    generated_answer = tokenizer.decode(answer[0], skip_special_tokens=True)

    if "Answer:" in generated_answer:
        extracted_answer = generated_answer.split("Answer:")[-1].strip()
    else:
        extracted_answer = generated_answer

    return extracted_answer

def baseline_oracle(retrieved_chunks, retrieved_indices, gold_index):
    indices_list = list(retrieved_indices)
    chunks_list = list(retrieved_chunks)

    if gold_index in indices_list:
        # put it at first (index 0)
        idx_in_list = indices_list.index(gold_index) 
        gold_idx = indices_list.pop(idx_in_list)
        gold_text = chunks_list.pop(idx_in_list)
        indices_list.insert(0, gold_idx)
        chunks_list.insert(0, gold_text)
    else:
        # delete the last one
        indices_list.pop(-1)
        chunks_list.pop(-1)
        # add the gold one at first
        indices_list.insert(0, gold_index)
        chunks_list.insert(0, retrieved_chunks[gold_index])
    
    return chunks_list, indices_list

def main():
    ds = load_data()
    model_name = "google/flan-t5-small"  # You can change this to any other model you want to test
    
    model, tokenizer, device = load_model(model_name, "seq2seq")  # Assuming it's a seq2seq model, change if needed
    queries = ds["test"]["query"]

    answers = {}

    for query in queries[:5]: # Limit to the first 5 queries for testing
        answer = baseline(model, tokenizer, query, device)
        answers[query] = f"Model Answer: {answer}; Real Answer: {ds['test']['answer'][queries.index(query)]}" # Store both model and real answers for comparison

    print("Final Answers:")
    for query, answer in answers.items():
        print(f"Query: {query}\nAnswer: {answer}\n")


    all_mini = 'sentence-transformers/all-MiniLM-L6-v2'
    retriever_model = SentenceTransformer(all_mini, device=device)

    answers_rag = {}

    for i in range(5):
        item = ds["test"][i]
        query = item["query"]
        candidate = item["candidate_chunks"]

        retrieved_texts, retrieved_indices = get_top_k_chunks(query, candidate, retriever_model, k=3)

        answer_rag = baseline_rag(model, tokenizer, query, retrieved_texts, device)
        answers_rag[query] = f"RAG Answer: {answer_rag};\nReal Answer: {item['short_answer']}"
        
    print("Final Answers RAG:")
    for query, answer in answers_rag.items():
        print(f"Query: {query}\nRAG Answer: {answer}\n")

    answers_oracle = {}
    for i in range(5):
        item = ds["test"][i]
        query = item["query"]
        candidate = item["candidate_chunks"]
        gold_indice = item["answer_pos"]
        short_answer = item["short_answer"]

        retrieved_texts, retrieved_indices = get_top_k_chunks(query, candidate, retriever_model, k=3)

        retrieved_texts_oracle, retrieved_indices_oracle = baseline_oracle(retrieved_texts, retrieved_indices, gold_indice)
        answer_oracle = baseline_rag(model, tokenizer, query, retrieved_texts_oracle, device)
        answers_oracle[query] = f"Oracle Answer: {answer_oracle};\nReal Answer: {short_answer}"
        
    print("Final Answers Oracle:")
    for query, answer in answers_oracle.items():
        print(f"Query: {query}\nOracle Answer: {answer}\n")



if __name__ == "__main__":
    main()