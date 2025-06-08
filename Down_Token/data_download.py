import os
from datasets import load_dataset

# Create directory if it doesn't exist
os.makedirs("../dataset", exist_ok=True)

# Load and save Wikipedia dump (English) - subset (~30 GB)
print("Loading Wikipedia dataset (subset)...")
wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train[:4%]")  # ~4% of full dataset
wiki_dataset.save_to_disk("dataset/wikipedia")
print(f"Wikipedia dataset saved. Size: {len(wiki_dataset)} examples")

# Load and save TinyStories - subset (~20 GB)
print("Loading TinyStories dataset (subset)...")
stories_dataset = load_dataset("roneneldan/TinyStories", split="train[:50%]")  # ~50% of full dataset
stories_dataset.save_to_disk("dataset/tinystories")
print(f"TinyStories dataset saved. Size: {len(stories_dataset)} examples")

# Load and save CC-News - subset (~10 GB)
print("Loading CC-News dataset (subset)...")
ccnews_dataset = load_dataset("cc_news", split="train[:8%]")  # ~8% of full dataset
ccnews_dataset.save_to_disk("dataset/cc_news")
print(f"CC-News dataset saved. Size: {len(ccnews_dataset)} examples")

# Load and save TWEETSUMM (Customer Service Chat) - FULL (~20 GB)
print("Loading TWEETSUMM dataset (FULL)...")
tweetsumm_dataset = load_dataset("tweetsumm", split="train")  # Customer service chat dataset
tweetsumm_dataset.save_to_disk("dataset/tweetsumm")
print(f"TWEETSUMM dataset saved. Size: {len(tweetsumm_dataset)} examples")

# Show total dataset stats
total_examples = len(wiki_dataset) + len(stories_dataset) + len(ccnews_dataset) + len(tweetsumm_dataset)
print(f"\nTotal examples across all datasets: {total_examples:,}")