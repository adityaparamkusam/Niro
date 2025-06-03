from __future__ import annotations
import argparse, torch, torch.nn.utils.prune as prune
from pathlib import Path
from transformers import AutoTokenizer
from rich import print as rprint
from model import GPTSmall

def prune_global(model, amt):
    to_prune=[(m,"weight") for m in model.modules() if isinstance(m,torch.nn.Linear)]
    prune.global_unstructured(to_prune, prune.L1Unstructured(), amount=amt)
    for m,_ in to_prune: prune.remove(m,"weight")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/pytorch_model.bin")
    ap.add_argument("--amount", type=float, default=0.2)
    ap.add_argument("--tokenizer", default="tokenizer")
    args=ap.parse_args()
    if not Path(args.ckpt).is_file(): rprint(f"[red]{args.ckpt} missing"); return
    tok=AutoTokenizer.from_pretrained(args.tokenizer)
    model=GPTSmall(len(tok), tok.pad_token_id)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    rprint(f"[yellow]Prune {args.amount*100:.0f}%[/yellow]"); prune_global(model,args.amount)
    model_q=torch.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8)
    out=Path(args.ckpt).with_suffix("_pruned_int8.pt"); torch.save(model_q.state_dict(),out)
    rprint(f"[green]Saved {out}[/green]")

if __name__=="__main__": main()
