import sentencepiece as spm, glob, argparse, textwrap, sys, os

ap = argparse.ArgumentParser()
ap.add_argument("--corpus_glob", default="Data/*.txt")
ap.add_argument("--vocab_size", type=int, default=32000)
ap.add_argument("--out", default="niro_bpe")
opt = ap.parse_args()

files = glob.glob(opt.corpus_glob)
if not files:
    sys.exit("No data shards; run download_data.sh first")

spm.SentencePieceTrainer.train(
    input=",".join(files),
    model_prefix=opt.out,
    vocab_size=opt.vocab_size,
    model_type="bpe",
    byte_fallback=True,
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    user_defined_symbols=["<|pad|>"])

print(textwrap.dedent(f"""
✅ Tokenizer ready:
  {opt.out}.model / {opt.out}.vocab
"""))