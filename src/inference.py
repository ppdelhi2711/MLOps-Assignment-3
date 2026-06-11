
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

# Auth 
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Model 
MODEL_REPO = os.environ.get("HF_MODEL", "pp2711/imdb-distilbert-sentiment")

# Input 
input_text = os.environ.get("INPUT_TEXT", "").strip()
if not input_text:
    print("ERROR: INPUT_TEXT environment variable is empty or not set.")
    sys.exit(1)

print(f"Model  : {MODEL_REPO}")
print(f"Input  : {input_text}")
print("-" * 50)

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.eval()

#Tokenise 
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=128,
)
# DistilBERT does not use token_type_ids — remove it if present
inputs.pop("token_type_ids", None)

# Predict
with torch.no_grad():
    logits = model(**inputs).logits

pred_id = logits.argmax(-1).item()
score = torch.softmax(logits, dim=-1)[0][pred_id].item()
label = model.config.id2label[pred_id]

# Output 
print(f"Label  : {label}")
print(f"Score  : {score:.4f}")
print("-" * 50)
print("Inference complete.")
