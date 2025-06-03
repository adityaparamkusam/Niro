from __future__ import annotations
import argparse, torch
from pathlib import Path
from transformers import AutoTokenizer
from rich import print as rprint
from model import GPTSmall

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/pytorch_model_pruned_int8.pt")
    ap.add_argument("--out",  default="niro_small_int8.onnx")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--seq_len",type=int,default=32)
    args=ap.parse_args()
    if not Path(args.ckpt).is_file(): rprint(f"[red]{args.ckpt} missing"); return
    tok=AutoTokenizer.from_pretrained(args.tokenizer)
    model=GPTSmall(len(tok), tok.pad_token_id)
    model.load_state_dict(torch.load(args.ckpt,map_location="cpu"),strict=False); model.eval()
    dummy=torch.randint(0,len(tok),(1,args.seq_len),dtype=torch.long)
    torch.onnx.export(
        model,dummy,args.out,opset_version=17,
        input_names=["input_ids"],output_names=["logits"],
        dynamic_axes={"input_ids":{0:"batch",1:"seq"},"logits":{0:"batch",1:"seq"}})
    rprint(f"[green]ONNX → {Path(args.out).resolve()}[/green]")

if __name__=="__main__": main()
