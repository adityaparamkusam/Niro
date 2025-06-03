import sentencepiece as spm, glob, numpy as np, tqdm, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--tok", default="niro_bpe.model")
ap.add_argument("--text_glob", default="Data/*.txt")
ap.add_argument("--out_dir", default="Data")
ap.add_argument("--seq_len", type=int, default=1024)
opt = ap.parse_args()

tok = spm.SentencePieceProcessor(model_file=opt.tok)
os.makedirs(opt.out_dir, exist_ok=True)

for fp in tqdm.tqdm(glob.glob(opt.text_glob)):
    ids = np.array(tok.encode(open(fp, encoding="utf-8").read()),
                   dtype=np.uint16)
    if ids.size < opt.seq_len: continue
    out = os.path.join(opt.out_dir, os.path.basename(fp)+".bin")
    ids.tofile(out)
print("✅ Bins written")