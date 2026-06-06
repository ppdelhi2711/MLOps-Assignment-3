## Select & Load a Model from Hugging Face, Train Multiple Versions on Kaggle & Track with W&B, Push Trained Model to Hugging Face Hub

import os
import json
import wandb
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login


VERSION = "v1"          
CONFIGS = {
    "v1": dict(epochs=3, lr=3e-5, batch=16),
    "v2": dict(epochs=4, lr=2e-5, batch=32),
}
cfg = CONFIGS[VERSION]


secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
HF_TOKEN = secrets.get_secret("HF_TOKEN")
login(token=HF_TOKEN)
wandb.login()


MODEL_NAME    = "distilbert-base-uncased"
HF_REPO       = "pp2711/imdb-distilbert-sentiment"   
WB_PROJECT    = "mlops-assignment3"
MAX_LEN       = 128
SEED          = 42


with open("/kaggle/input/mlops-a3/id2label.json") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)


train_df = pd.read_csv("/kaggle/input/mlops-a3/train.csv")
test_df  = pd.read_csv("/kaggle/input/mlops-a3/test.csv")


train_df = train_df.sample(5000, random_state=SEED).reset_index(drop=True)
test_df  = test_df.sample(1000,  random_state=SEED).reset_index(drop=True)

train_ds = Dataset.from_pandas(train_df)
test_ds  = Dataset.from_pandas(test_df)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize,  batched=True)
train_ds = train_ds.rename_column("label", "labels")
test_ds  = test_ds.rename_column("label", "labels")
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)


wandb.init( 
entity="prateekpriyadarshi-iit-jodhpur",
    project=WB_PROJECT,
    name=f"run-{VERSION}",
    config={
        "model":         MODEL_NAME,
        "epochs":        cfg["epochs"],
        "batch_size":    cfg["batch"],
        "learning_rate": cfg["lr"],
        "max_len":       MAX_LEN,
        "dataset":       "imdb-5k",
        "version":       VERSION,
        "platform":      "Kaggle",
    },
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }


training_args = TrainingArguments(
    output_dir                  = f"./results-{VERSION}",
    num_train_epochs            = cfg["epochs"],
    per_device_train_batch_size = cfg["batch"],
    per_device_eval_batch_size  = 64,
    learning_rate               = cfg["lr"],
    weight_decay                = 0.01,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1",
    report_to                   = "wandb",
    run_name                    = f"run-{VERSION}",
    seed                        = SEED,
    fp16                        = True,        
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = test_ds,
    compute_metrics = compute_metrics,
)


print(f"\n{'='*50}")
print(f"  Training {VERSION}: epochs={cfg['epochs']}, lr={cfg['lr']}, batch={cfg['batch']}")
print(f"{'='*50}\n")

trainer.train()


results = trainer.evaluate()
print("\nEvaluation results:", results)


model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)
print(f"\nModel pushed to https://huggingface.co/{HF_REPO}")


wandb.run.summary["huggingface_model"] = f"https://huggingface.co/{HF_REPO}"
wandb.run.summary["eval_accuracy"]     = results["eval_accuracy"]
wandb.run.summary["eval_f1"]           = results["eval_f1"]

wandb.finish()
print("\nTraining complete.")
