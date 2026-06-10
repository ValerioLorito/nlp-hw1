import os
import json
import pandas as pd
import sys
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from wikidata_utils import get_wikidata_entity
from src.data_loader import load_data

def patch_jsonl_file(filepath, wikidata_cache):
    if not os.path.exists(filepath):
        print(f"  -> File not found, skipping: {filepath}")
        return

    filename = os.path.basename(filepath)
    records = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    # Usiamo tqdm con il nome del file per capire cosa sta processando
    for record in tqdm(records, desc=f"Patching {filename}"):
        query_id = record.get("query_id")
        wikidata_context = wikidata_cache.get(query_id)
        
        if not wikidata_context:
            continue # Salta se per qualche motivo non abbiamo i dati di questa query

        augmented_prompt = record.get("augmented_prompt", "")
        
        if "Answer:" in augmented_prompt:
            parts = augmented_prompt.split("Answer:")
            new_prompt = f"{parts[0].strip()}\n{wikidata_context}\nAnswer:"
            record["augmented_prompt"] = new_prompt
            
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def patch_excel_file(filepath, wikidata_cache):
    if not os.path.exists(filepath):
        print(f"  -> File not found, skipping: {filepath}")
        return

    filename = os.path.basename(filepath)
    df = pd.read_excel(filepath, sheet_name="Jugements")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Patching {filename}"):
        query_id = row['query_id']
        wikidata_context = wikidata_cache.get(query_id)
        
        if not wikidata_context:
            continue
            
        augmented_prompt = str(row['augmented_prompt'])
        
        if "Answer:" in augmented_prompt:
            parts = augmented_prompt.split("Answer:")
            new_prompt = f"{parts[0].strip()}\n{wikidata_context}\nAnswer:"
            df.at[index, 'augmented_prompt'] = new_prompt

    df.to_excel(filepath, sheet_name="Jugements", index=False)


def main():
    # 1. Carichiamo i dati una sola volta
    print("Loading dataset...")
    ds = load_data()
    id_map = {item["query_id"]: item["wikidata_id"] for item in ds["test"]}
    
    # 2. Creiamo la cache scaricando i dati da Wikidata (Fatto 1 sola volta per tutte le query!)
    wikidata_cache = {}
    print("Pre-fetching Wikidata information for all queries (this will happen only once)...")
    for query_id, wikidata_id in tqdm(id_map.items(), desc="Caching Wikidata Info"):
        if wikidata_id:
            # Scarichiamo e salviamo il testo nella cache
            wikidata_cache[query_id] = get_wikidata_entity([wikidata_id], id_only=False)

    print("\n--- Cache built successfully! Starting file patching ---\n")

    files_to_patch = [
        "rag/answers/Its_always_loss-test-flan-t5-large-RAG.jsonl",
        "rag/answers/Its_always_loss-test-llama-3.2-1b-instruct-RAG.jsonl",
        "rag/answers/Its_always_loss-test-flan-t5-large-RAG_scored.jsonl",
        "rag/answers/Its_always_loss-test-llama-3.2-1b-instruct-RAG_scored.jsonl",
        "rag/answers/Its_always_loss-test-flan-t5-large-Oracle.jsonl",
        "rag/answers/Its_always_loss-test-flan-t5-large-Oracle_scored.jsonl",
        "rag/answers/Its_always_loss-test-llama-3.2-1b-instruct-Oracle.jsonl",
        "rag/answers/Its_always_loss-test-llama-3.2-1b-instruct-Oracle_scored.jsonl",
        "rag/evaluation/Its_always_loss-evaluation-flan-t5-large-RAG-JUDGE.jsonl"
    ]

    # 3. Passiamo la cache a tutti i file (il patching ora sarà quasi istantaneo)
    for f in files_to_patch:
        patch_jsonl_file(f, wikidata_cache)

    excel_file = "rag/evaluation/Annotations-flan-t5-large.xlsx"
    patch_excel_file(excel_file, wikidata_cache)
    
    print("\nAll files successfully patched!")

if __name__ == "__main__":
    main()