from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tok.add_special_tokens({"pad_token": "[PAD]", "eos_token": "</s>"})
tok.save_pretrained("tokenizer")
print("Tokenizer saved → tokenizer/")