import sys
import os
import json
from tqdm import tqdm
from evaluation import evaluate_all
from output_files import generate_jsonl_file

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.data_loader import load_data
from utils import load_model
from wikidata_utils import get_wikidata_entity

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
        pad_token_id=tokenizer.eos_token_id
    )

    generated_answer = tokenizer.decode(answer[0], skip_special_tokens=True)

    if "Answer:" in generated_answer:
        extracted_answer = generated_answer.split("Answer:")[-1].strip()
    else:
        extracted_answer = generated_answer

    return extracted_answer


def get_top_k_chunks(query_id, jsonl_path, candidate_chunks, k=3):  
    top_k_indices = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if query_id in data: 
                top_k_indices = data[query_id][:k]
                break
    
    top_k_chunks = []
    for i in top_k_indices:
        top_k_chunks.append(candidate_chunks[i])
    
    return top_k_chunks, top_k_indices


def rag(model, tokenizer, query, wikidata_info, retrieved_passages, device):
    context = []
    for index, passage in enumerate(retrieved_passages):
        formatted_passage = f"Document {index+1}: {passage}"
        context.append(formatted_passage)
    
    if wikidata_info:
        context.append(wikidata_info)

    context = "\n---\n".join(context) # concatenation of retrieved passages

    prompt = (f"You are an expert in question answering. Given a set of retrieved documents, and information about a Wikidata entity, provide a concise and correct answer to the question.\n\n"
             f"Context: {context}\n"
             f"Question: {query}\n" # The question is being inserted after the context to avoid "Lost in the middle" issues
             f"Answer: ")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length= tokenizer.model_max_length if tokenizer.model_max_length >= 1536 else 1536).to(device)

    truncated_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    print(f"\n--- FINAL PROMPT SENT TO THE MODEL {model.__class__.__name__} ---")
    print(truncated_prompt)
    print("-------------------------------------------------------------------\n")


    answer = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7, 
        top_p=0.9, 
        do_sample=True,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_answer = tokenizer.decode(answer[0], skip_special_tokens=True)

    if "Answer:" in generated_answer:
        extracted_answer = generated_answer.split("Answer:")[-1].strip()
    else:
        extracted_answer = generated_answer

    return extracted_answer, truncated_prompt

def oracle(retrieved_chunks, retrieved_indices, gold_index, candidates):
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
        chunks_list.insert(0, candidates[gold_index])
    
    return chunks_list, indices_list

def main():
    ds = load_data()

    queries = ds["test"]["query"]
    wikidata_ids = ds["test"]["wikidata_id"]

    t5_model = "google/flan-t5-large"  # You can change this to any other model you want to test
    t5_model, t5_tokenizer, t5_device = load_model(t5_model, "seq2seq") # Flan-T5 is a seq2seq model, so we specify "seq2seq" here
   
    llama_model = "meta-llama/Llama-3.2-1b-instruct"  # You can change this to any other model you want to test
    llama_model, llama_tokenizer, llama_device = load_model(llama_model, "causal") # LLaMA is a causal model, so we specify "causal" here

    answers = {}
    scores= {}

    # Baseline pipeline
    for query in tqdm(queries, desc="Baseline Pipeline Processing"): # Limit to the first 5 queries for testing
        item = ds["test"][queries.index(query)]
        short_answer = item["short_answer"]

        t5_answer = baseline(t5_model, t5_tokenizer, query, t5_device)
        llama_answer = baseline(llama_model, llama_tokenizer, query, llama_device)

        answers[query] = f"1st Model Answer: {t5_answer}\n2nd Model Answer: {llama_answer}\nReal Answer: {short_answer}" # Store real answer
        t5_scores = evaluate_all(t5_answer, short_answer)
        llama_scores = evaluate_all(llama_answer, short_answer)
        scores[query] = f"1st Model Scores: {t5_scores}\n2nd Model Scores: {llama_scores}"

    print("------------Final Answers (Baseline):------------")
    for query, answer in answers.items():
        print(f"Query: {query}\n{answer}\n{scores[query]}\n")

    # RAG and Oracle pipeline
    PREDICTIONS_DIR = os.path.join(parent_dir, "predictions")
    all_mini_jsonl = os.path.join(PREDICTIONS_DIR, "test", "Its_always_loss-test-all-miniLM-L6-v2-2-mnr-cosine.jsonl")
    answers_rag = {}
    scores_rag = {}
    answers_oracle = {} 
    scores_oracle = {}
    t5_rag_all_results = []
    llama_rag_all_results = []
    t5_oracle_all_results = []
    llama_oracle_all_results = []

    for query in tqdm(queries, desc="Baseline Pipeline Procesing"): # Limit to the first 5 queries for testing
        item = ds["test"][queries.index(query)]
        query = item["query"]
        query_id = item["query_id"]
        candidate = item["candidate_chunks"]
        gold_index = item["answer_pos"]
        short_answer = item["short_answer"]
        wikidata_id = item["wikidata_id"]

        wikidata_info = get_wikidata_entity(wikidata_id)

        #print(f"\n--- WIKIDATA INFORMATION RETRIEVED FOR ENTITY {wikidata_id} ---")
        #print(wikidata_info)

        # RAG pipeline
        retrieved_chunks, retrieved_indices = get_top_k_chunks(query_id, all_mini_jsonl, candidate, k=3)
        
        t5_answer_rag, t5_augmented_prompt = rag(t5_model, t5_tokenizer, query, wikidata_id, retrieved_chunks, t5_device)
        llama_answer_rag, llama_augmented_prompt = rag(llama_model, llama_tokenizer, query, wikidata_id, retrieved_chunks, llama_device)

        answers_rag[query] = f"1st Model RAG Answer: {t5_answer_rag}\n2nd Model RAG Answer: {llama_answer_rag}\nReal Answer: {short_answer}"

        t5_scores_rag = evaluate_all(t5_answer_rag, short_answer)
        llama_scores_rag = evaluate_all(llama_answer_rag, short_answer)

        scores_rag[query] = f"1st Model Scores: {t5_scores_rag}\n2nd Model Scores: {llama_scores_rag}"

        t5_rag_all_results.append({
            "query_id": query_id,
            "retrieved_chunks": retrieved_indices,
            "augmented_prompt": t5_augmented_prompt,
            "generated_answer": t5_answer_rag
        })

        llama_rag_all_results.append({
            "query_id": query_id,
            "retrieved_chunks": retrieved_indices,
            "augmented_prompt": llama_augmented_prompt,
            "generated_answer": llama_answer_rag
        })

        # Oracle pipeline
        retrieved_chunks_oracle, retrieved_indices_oracle = oracle(retrieved_chunks, retrieved_indices, gold_index, candidate)
        
        t5_answer_oracle, t5_oracle_augmented_prompt = rag(t5_model, t5_tokenizer, query, wikidata_id, retrieved_chunks_oracle, t5_device)
        llama_answer_oracle, llama_oracle_augmented_prompt = rag(llama_model, llama_tokenizer, query, wikidata_id, retrieved_chunks_oracle, llama_device)

        answers_oracle[query] = f"1st Model Oracle Answer: {t5_answer_oracle}\n2nd Model Oracle Answer: {llama_answer_oracle}\nReal Answer: {short_answer}"
        
        t5_scores_oracle = evaluate_all(t5_answer_oracle, short_answer)
        llama_scores_oracle = evaluate_all(llama_answer_oracle, short_answer)

        scores_oracle[query] = f"1st Model Scores: {t5_scores_oracle}\n2nd Model Scores: {llama_scores_oracle}"

        t5_oracle_all_results.append({
            "query_id": query_id,
            "retrieved_chunks": retrieved_indices_oracle,
            "augmented_prompt": t5_oracle_augmented_prompt,
            "generated_answer": t5_answer_oracle
        })

        llama_oracle_all_results.append({
            "query_id": query_id,
            "retrieved_chunks": retrieved_indices_oracle,
            "augmented_prompt": llama_oracle_augmented_prompt,
            "generated_answer": llama_answer_oracle
        })

    print("------------Final Answers (RAG):--------------")
    for query, answer in answers_rag.items():
        print(f"Query: {query}\n{answer}\n{scores_rag[query]}\n")
        
    print("------------Final Answers (Oracle):---------------")
    for query, answer in answers_oracle.items():
        print(f"Query: {query}\n{answer}\n{scores_oracle[query]}\n")

    generate_jsonl_file(t5_rag_all_results, "all-test", "flan-t5-large", "RAG", "generated_responses")
    generate_jsonl_file(llama_rag_all_results, "all-test", "Llama-3.2-1b-instruct", "RAG", "generated_responses")
    generate_jsonl_file(t5_oracle_all_results, "all-test", "flan-t5-large", "Oracle", "generated_responses")
    generate_jsonl_file(llama_oracle_all_results, "all-test", "Llama-3.2-1b-instruct", "Oracle", "generated_responses")
    


if __name__ == "__main__":
    main()