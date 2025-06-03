from __future__ import annotations
import argparse, math
from pathlib import Path
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from rich.progress import track

from model import GPTSmall

def loader(dir_, tok, bs, shuffle):
    ds = load_from_disk(dir_)
    coll = DataCollatorForLanguageModeling(tok, mlm=False)
    return torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=shuffle, collate_fn=coll,
        pin_memory=torch.cuda.is_available())

def val_ppl(model, dl, tok):
    model.eval(); loss=toks=0
    with torch.no_grad():
        for b in dl:
            l = cross_entropy(
                model(b["input_ids"]).view(-1, len(tok)),
                b["labels"].view(-1),
                ignore_index=-100,
                reduction="sum")
            loss += l.item()
            toks += (b["labels"] != -100).sum().item()
    return math.exp(loss / toks)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="viro_dummy_data/qa_tokenized/train")
    ap.add_argument("--valid_dir", default="viro_dummy_data/qa_tokenized/validation")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--checkpoint", default="ckpts/pytorch_model.bin")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs",type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    args=ap.parse_args()

    accel=Accelerator(); tok=AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    train_dl=loader(args.train_dir, tok, args.batch, True)
    val_dl  =loader(args.valid_dir, tok, args.batch, False)

    if not Path(args.checkpoint).is_file():
        accel.print(f"Checkpoint {args.checkpoint} not found"); return
    model=GPTSmall(len(tok), tok.pad_token_id)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)

    opt=AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched=OneCycleLR(opt, max_lr=args.lr, total_steps=len(train_dl)*args.epochs)

    model,opt,sched,train_dl,val_dl=accel.prepare(model,opt,sched,train_dl,val_dl)
    for ep in range(1,args.epochs+1):
        model.train()
        for b in track(train_dl, description=f"[cyan]Ep {ep}"):
            opt.zero_grad()
            loss=cross_entropy(
                model(b["input_ids"]).view(-1, len(tok)),
                b["labels"].view(-1), ignore_index=-100)
            accel.backward(loss); opt.step(); sched.step()
        ppl=val_ppl(model,val_dl,tok)
        accel.print(f"Epoch {ep} | val PPL {ppl:.2f}")
        if accel.is_main_process: accel.save_state("ckpts")

if __name__=="__main__":
    main()
