from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses, evaluation
from sentence_transformers.evaluation import BinaryClassificationEvaluator, RerankingEvaluator

from src.data_loader import load_data
from src.strategies import create_sentence_pairs, create_batches, create_samples

def train_args(model, strategy, learning_rate=2e-5, epochs=3, warmup_steps=0.1) -> SentenceTransformerTrainingArguments:
  return SentenceTransformerTrainingArguments(
    output_dir="models/" + model + "-" + strategy,
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    warmup_steps=warmup_steps,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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
    ds = load_data()

    strategies = {
        "pairs": {
            "train_set": create_sentence_pairs(ds["train"]),
            "dev_set": create_sentence_pairs(ds["dev"]),
        },
        "mnr": {
            "train_set": create_batches(ds["train"]),
            "dev_set": create_batches(ds["dev"]),
        }
    }

    models = {
        "distilbert_pairs": {
            "model_name": "distilbert/distilbert-base-uncased",
            "strategy": "pairs",
            "loss": losses.ContrastiveLoss,
            **strategies["pairs"]
        },
        "all-mini_pairs": {
            "model_name": "sentence-transformers/all-miniLM-L6-v2",
            "strategy": "pairs",
            "loss": losses.ContrastiveLoss,
            **strategies["pairs"]
        },
        "distilbert_mnr": {
            "model_name": "distilbert/distilbert-base-uncased",
            "strategy": "mnr",
            "loss": losses.MultipleNegativesRankingLoss,
            **strategies["mnr"]
        },
        "all-mini_mnr": {
            "model_name": "sentence-transformers/all-miniLM-L6-v2",
            "strategy": "mnr",
            "loss": losses.MultipleNegativesRankingLoss,
            **strategies["mnr"]
        },
    }

    for model_key, model_item in models.items():
        if model_item["strategy"] == "pairs":
                model_item["dev_evaluator"] = BinaryClassificationEvaluator(
                sentences1=model_item["dev_set"]["query"],
                sentences2=model_item["dev_set"]["candidate"],
                labels=model_item["dev_set"]["label"],
                name=model_item["model_name"] + "-pairs-dev",
                write_csv=False
            )

        elif model_item["strategy"] == "mnr":
            model_item["dev_evaluator"] = RerankingEvaluator(
                samples=create_samples(model_item["dev_set"]),
                name=model_key + "-mnr-dev",
                write_csv=False
            )

        model = SentenceTransformer(model_item["model_name"])

        loss = model_item["loss"](model)

        args = train_args(model_item["model_name"], model_item["strategy"])

        trainer = sentence_transformer_trainer(model, args, model_item["train_set"], model_item["dev_set"], loss, model_item["dev_evaluator"])

        trainer.train()
        model_item["model"].save("models/" + model_item["name"] + "-" + model_item["strategy"])

if __name__ == "__main__":
    main()