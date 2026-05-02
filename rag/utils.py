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