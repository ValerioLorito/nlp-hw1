import gc

import torch
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses, evaluation
from sentence_transformers.evaluation import BinaryClassificationEvaluator, RerankingEvaluator

from src.data_loader import load_data
from src.baseline_test import embedding
from src.strategies import create_sentence_pairs, create_batches, create_samples

def train_args(model, strategy, learning_rate=2e-5, epochs=3, warmup_steps=0.1) -> SentenceTransformerTrainingArguments:
  return SentenceTransformerTrainingArguments(
    output_dir="models/" + model + "-" + strategy,
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    warmup_steps=warmup_steps,
    per_device_train_batch_size=8, # Set to 16 if Windows/Linux or GPU with more memory is available
    per_device_eval_batch_size=8, # Set to 16 if Windows/Linux or GPU with more memory is available
    gradient_accumulation_steps=2, # Compensates for the small batch size by accumulating gradients over multiple steps
    dataloader_pin_memory=False, # Avoids memory issues on GPU (only for Macbook)
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    run_name=model
  )

def sentence_transformer_trainer(model, args, train_set, dev_set, loss, evaluator) -> SentenceTransformerTrainer:
  return SentenceTransformerTrainer(
      model=model,
      args=args,
      train_dataset=train_set,
      eval_dataset=dev_set,
      loss=loss,
      evaluator=evaluator
  )

def main():
    ds = load_data() # Load the dataset using our data loader function

    # We define a dictionary of models to train, with their corresponding training strategy and loss function.
    models = {
        "distilbert_pairs": {
            "model_name": "distilbert/distilbert-base-uncased",
            "strategy": "pairs",
            "loss": losses.ContrastiveLoss,
            "strategy": "pairs",
        },
        "all-mini_pairs": {
            "model_name": "sentence-transformers/all-miniLM-L6-v2",
            "strategy": "pairs",
            "loss": losses.ContrastiveLoss,
            "strategy": "pairs",
        },
        "distilbert_mnr": {
            "model_name": "distilbert/distilbert-base-uncased",
            "strategy": "mnr",
            "loss": losses.MultipleNegativesRankingLoss,
            "strategy": "mnr",
        },
        "all-mini_mnr": {
            "model_name": "sentence-transformers/all-miniLM-L6-v2",
            "strategy": "mnr",
            "loss": losses.MultipleNegativesRankingLoss,
            "strategy": "mnr",
        },
    }

    # For each model in the dictionary, we create the appropriate training and evaluation structure
    for _, model_item in models.items():
        
        model = SentenceTransformer(model_item["model_name"]) # Loading the pre-trained model

        tokenizer = AutoTokenizer.from_pretrained(model_item["model_name"]) # Loading the corresponding tokenizer

        train_query_embeddings, train_candidates_embeddings = embedding(ds["train"]["query"], ds["train"]["candidate_chunks"], tokenizer, model)
        dev_query_embeddings, dev_candidates_embeddings = embedding(ds["dev"]["query"], ds["dev"]["candidate_chunks"], tokenizer, model)

        if model_item["strategy"] == "pairs": # We create sentence pairs for the contrastive loss strategy, and a BinaryClassificationEvaluator for evaluation
            ds_train = create_sentence_pairs(ds["train"], train_query_embeddings, train_candidates_embeddings)
            ds_dev = create_sentence_pairs(ds["dev"], dev_query_embeddings, dev_candidates_embeddings)
            dev_evaluator = BinaryClassificationEvaluator(
                sentences1=ds_dev["sentence1"],
                sentences2=ds_dev["sentence2"],
                labels=ds_dev["label"],
                name=model_item["model_name"] + "-pairs-dev",
                write_csv=False
            )

        elif model_item["strategy"] == "mnr": # We create batches with multiple negatives for the Multiple Negatives Ranking Loss strategy, and a RerankingEvaluator for evaluation
            ds_train = create_batches(ds["train"], train_query_embeddings, train_candidates_embeddings)
            ds_dev = create_batches(ds["dev"], dev_query_embeddings, dev_candidates_embeddings)
            dev_evaluator = RerankingEvaluator(
                samples=create_samples(ds_dev),
                name=model_item["model_name"] + "-mnr-dev",
                write_csv=False
            )

        loss = model_item["loss"](model) # Inizialiting the loss function with the model

        args = train_args(model_item["model_name"], model_item["strategy"]) # Getting the training arguments for the model and strategy

        trainer = sentence_transformer_trainer(model, args, ds_train, ds_dev, loss, dev_evaluator) # Creating the trainer for the model and strategy

        # Training and saving the model
        trainer.train()
        model.save_pretrained(args.output_dir)

        del model
        del trainer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    main()