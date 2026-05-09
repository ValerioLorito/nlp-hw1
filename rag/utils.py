import requests
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

def load_model(model, model_type):
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(model)
    elif model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model)
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)

    return model, tokenizer, device

def get_wikidata_entity(wikidata_id, language="en"):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "format": "json",
        "languages": language,
        "props": "labels|descriptions|aliases"
    }
    response = requests.get(url, params=params)
    data = response.json()

    entity = data.get("entities", {}).get(wikidata_id, {})
    label = entity.get("labels", {}).get(language, {}).get("value", "Unknown label")
    description = entity.get("descriptions", {}).get(language, {}).get("value", "Unknown description")
    aliases = entity.get("aliases", {}).get(language, []).get("value", None)

    wikidata_entity_info = f"Wikidata Information: {label} - {description}; Aliases: {', '.join(aliases) if aliases else 'None'}"

    if response.status_code == 200:
        return wikidata_entity_info
    else:
        return None
