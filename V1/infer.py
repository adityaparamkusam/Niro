from __future__ import annotations
import argparse, os, torch
from transformers import AutoTokenizer
from rich import print as rprint
from model import GPTSmall

def load_torch(ckpt, tok):
    device="cuda" if torch.cuda.is_available() else "cpu"
    m=GPTSmall(len(tok), tok.pad_token_id); m.load_state_dict(torch.load(ckpt,map_location=device),strict=False)
    m.to(device).eval(); return m,device

def load_onnx(ckpt):
    import onnxruntime as ort
    providers=["CUDAExecutionProvider","CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    sess=ort.InferenceSession(ckpt, providers=providers)
    return sess

def gen_torch(model,ids,top_k=40,temp=1.0,max_new=64,stop=None):
    device=ids.device
    for _ in range(max_new):
        logits=model(ids)[:,-1,:]/temp
        next_id=torch.topk(logits,top_k).indices[0,torch.randint(0,top_k,(1,))]
        ids=torch.cat([ids,next_id.view(1,1).to(device)],dim=-1)
        if stop and next_id.item()==stop: break
    return ids

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--backend",choices=["torch","onnx"],default="torch")
    ap.add_argument("--checkpoint",default="ckpts/pytorch_model_pruned_int8.pt")
    ap.add_argument("--tokenizer",default="tokenizer")
    args=ap.parse_args()
    if not os.path.isfile(args.checkpoint): rprint("[red]Checkpoint missing"); return
    tok=AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if args.backend=="torch":
        model,device=load_torch(args.checkpoint,tok)
        run=lambda x: model(x.to(device))
    else:
        sess=load_onnx(args.checkpoint)
        run=lambda x: torch.from_numpy(sess.run(None,{"input_ids":x.cpu().numpy()})[0])
        device="cpu"
    rprint("[green]NIRO ready – type 'exit'[/green]")
    while True:
        try:
            text=input("You: ").strip()
            if text.lower() in {"exit","quit"}: break
            ids=tok(text,return_tensors="pt").input_ids.to(device)
            out=gen_torch(run,ids,max_new=64,stop=tok.eos_token_id)
            reply=tok.decode(out[0],skip_special_tokens=True)[len(text):].strip()
            rprint(f"[bold yellow]NIRO:[/bold yellow] {reply}")
        except KeyboardInterrupt: break

if __name__=="__main__": main()
