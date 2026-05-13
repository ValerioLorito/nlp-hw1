import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

def load_model(model, model_type):
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    if device == 'cuda':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, # Accelerate computation
            bnb_4bit_quant_type="nf4",            # best compression quality
            bnb_4bit_use_double_quant=True        # VRAM economy
        )
        if model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb_config, device_map="auto")
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model, quantization_config=bnb_config, device_map="auto")
        device = next(model.parameters()).device

    else:
        if model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(model)
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model)
        model.to(device)
    
    return model, tokenizer, device