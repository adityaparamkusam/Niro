from transformers import AutoTokenizer
from datasets import load_from_disk
import os

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Tokenization function
def tokenize_function(examples):
    texts = examples.get("text", examples.get("dialogue", ""))  # Handle 'text' or 'dialogue' fields
    texts = texts if isinstance(texts, list) else [texts]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=128,  # Shorter length for small model efficiency
        padding=False,
        return_attention_mask=True,
        return_tensors=None
    )
    return tokenized


# Get all dataset folders in 'dataset' directory
dataset_dir = "../dataset"
dataset_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f)) and not f.endswith("_tokenized")]

# Process each dataset
for dataset in dataset_folders:
    print(f"Loading {dataset} dataset...")
    ds = load_from_disk(f"dataset/{dataset}")

    print(f"Tokenizing {dataset} dataset...")
    tokenized_ds = ds.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
        desc=f"Tokenizing {dataset}"
    )

    print(f"Saving tokenized {dataset}...")
    tokenized_ds.save_to_disk(f"dataset/{dataset}_tokenized")
    print(f"{dataset} tokenized: {len(tokenized_ds)} examples")