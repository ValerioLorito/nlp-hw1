import os
import json
import sys
import time
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.data_loader import load_data
from wikidata_utils import get_wikidata_entity

CACHE_FILE = "rag/all-test/wikidata_cache.json" # Percorso dove salvare il file

def build_cache():
    print("Caricamento dataset...")
    ds = load_data()
    
    # 1. Raccogliamo tutti i wikidata_id unici dal test set
    unique_wikidata_ids = set()
    for item in ds["test"]:
        if item["wikidata_id"]:
            unique_wikidata_ids.add(item["wikidata_id"])
            
    # 2. Carichiamo la cache esistente se il programma si era interrotto
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Trovata cache esistente con {len(cache)} elementi.")
        
    ids_to_fetch = [wid for wid in unique_wikidata_ids if wid not in cache]
    print(f"Rimangono da scaricare {len(ids_to_fetch)} entità Wikidata.")

    if not ids_to_fetch:
        print("La cache è già completa al 100%!")
        return

    # 3. Download con salvataggio incrementale (anti-crash)
    for wid in tqdm(ids_to_fetch, desc="Scaricando dati da Wikidata"):
        try:
            # Scarica le informazioni
            info = get_wikidata_entity([wid], id_only=False)
            cache[wid] = info
            
            # Salva su disco ogni 10 iterazioni. Se l'API crasha, perdi al massimo 9 query
            if len(cache) % 10 == 0:
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=4)
                    
            # Pausa di 0.1 secondi per non far arrabbiare l'API di Wikidata (Rate Limiting)
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"\n[!] Errore critico con l'ID {wid}: {e}")
            print("Salvataggio di emergenza della cache. Rilancia lo script per riprendere.")
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=4)
            sys.exit(1)

    # Salvataggio finale
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)
    print("\nDownload completato! File salvato in:", CACHE_FILE)

if __name__ == "__main__":
    build_cache()