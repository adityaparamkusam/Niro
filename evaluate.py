import torch, sentencepiece as spm, glob, math
from model import GPTSmall
from tqdm import tqdm
import evaluate as hf_eval

tok=spm.SentencePieceProcessor(model_file="niro_bpe.model")
model=GPTSmall(tok.vocab_size(),tok.pad_id()); model.load_state_dict(
    torch.load("checkpoints/niro_final.pt",map_location="cpu"))
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

def perplexity(files):
    nll,tokens=0,0
    with torch.no_grad():
        for fp in tqdm(files):
            ids=torch.tensor(tok.encode(open(fp).read(),out_type=int))
            if ids.size(0)<2: continue
            out=model(ids[:-1].unsqueeze(0))
            nll+=torch.nn.functional.cross_entropy(out.squeeze(0),ids[1:],reduction="sum").item()
            tokens+=ids.size(0)-1
    return math.exp(nll/tokens)

def bleu(files):
    refs,preds=[],[]
    gen=hf_eval.load("bleu")
    for fp in tqdm(files):
        text=open(fp).read(); pred=tok.decode(model.generate(
            torch.tensor([[tok.bos_id()]]),max_new=64)[0].tolist())
        refs.append([text.split()]); preds.append(pred.split())
    return gen.compute(predictions=preds,references=refs)['bleu']

held=glob.glob("Data/heldout/*.txt")
print(f"PPL:  {perplexity(held):.2f}")
print(f"BLEU: {bleu(held):.2f}")