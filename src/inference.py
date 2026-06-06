"""
Task 7: Inference script
Called by GitHub Actions inference workflow.

Environment variables:
  HF_TOKEN    – Hugging Face read token (from GitHub Secrets)
  INPUT_TEXT  – Text to classify (from workflow_dispatch input)
  HF_MODEL    – (optional) override the model repo
"""

import os
import sys
from transformers import pipeline
from huggingface_hub import login

# ── Auth ──────────────────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_REPO = os.environ.get("HF_MODEL", "pp2711/imdb-distilbert-sentiment")

# ── Input ─────────────────────────────────────────────────────────────────────
input_text = os.environ.get("INPUT_TEXT", "").strip()
if not input_text:
    print("ERROR: INPUT_TEXT environment variable is empty or not set.")
    sys.exit(1)

print(f"Model  : {MODEL_REPO}")
print(f"Input  : {input_text}")
print("-" * 50)

# ── Run inference ─────────────────────────────────────────────────────────────
clf = pipeline(
    task="text-classification",
    model=MODEL_REPO,
    tokenizer=MODEL_REPO,
    truncation=True,
    max_length=128,
)

result = clf(input_text)[0]

print(f"Label  : {result['label']}")
print(f"Score  : {result['score']:.4f}")
print("-" * 50)
print("Inference complete.")
