"""
Data Preparation & Normalisation
Dataset : IMDb Movie Reviews
Output  : train.csv, test.csv, id2label.json
"""

import os
import re
import json
import pandas as pd
from datasets import load_dataset
from collections import Counter


SAVE_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_LENGTH = 512          # DistilBERT context window
SEED       = 42


print("Loading IMDb dataset from Hugging Face...")
raw = load_dataset("imdb")

train_raw = raw["train"].to_pandas()   # 25 000 rows
test_raw  = raw["test"].to_pandas()    #  25 000 rows

print(f"Raw train size : {len(train_raw)}")
print(f"Raw test  size : {len(test_raw)}")
print(f"Class distribution (train): {Counter(train_raw['label'])}")


def clean_text(text: str) -> str:
    ## For the Data Preparation the below points to be noted.
    """
    - Strip HTML tags (IMDb reviews contain <br /> tags)
    - Collapse multiple whitespace into a single space
    - Strip leading/trailing whitespace
    - Truncate to MAX_LENGTH tokens (character-level proxy: 4 chars ≈ 1 token)
    """
    
    text = re.sub(r"<[^>]+>", " ", text)          
    text = re.sub(r"\s+", " ", text).strip()       
    text = text[: MAX_LENGTH * 4]                  
    return text

print("\nCleaning text")
train_raw["text"] = train_raw["text"].apply(clean_text)
test_raw["text"]  = test_raw["text"].apply(clean_text)

# Remove duplicates
before = len(train_raw)
train_raw = train_raw.drop_duplicates(subset="text")
print(f"Duplicates removed (train): {before - len(train_raw)}")

# Check for missing values
print(f"Missing values (train): {train_raw.isnull().sum().to_dict()}")
train_raw = train_raw.dropna(subset=["text", "label"])
test_raw  = test_raw.dropna(subset=["text", "label"])

# Label encoding 
# IMDb: 0 = negative, 1 = positive (already numeric — just document it)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {v: k for k, v in id2label.items()}

id2label_path = os.path.join(SAVE_DIR, "..", "id2label.json")
with open(id2label_path, "w") as f:
    json.dump(id2label, f, indent=2)
print(f"\nSaved id2label.json → {id2label_path}")

# Save prepared splits 
train_path = os.path.join(SAVE_DIR, "train.csv")
test_path  = os.path.join(SAVE_DIR, "test.csv")

train_raw[["text", "label"]].to_csv(train_path, index=False)
test_raw[["text",  "label"]].to_csv(test_path,  index=False)

print(f"Saved train.csv ({len(train_raw)} rows) → {train_path}")
print(f"Saved test.csv  ({len(test_raw)}  rows) → {test_path}")
print("\nData preparation complete.")
