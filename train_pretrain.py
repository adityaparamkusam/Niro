from __future__ import annotations
import argparse, math, time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.progress import track

from dataset import get_dataloader
from model import GPTSmall

def evaluate(model, loader, tok, accel):
    model.eval(); loss_sum=tokens=0
    with torch.no_grad():
        for b in loader:
            logits = model(b["input_ids"])
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)),
                b["labels"].view(-1),
                ignore_index=-100,
                reduction="sum")
            loss_sum += accel.gather(loss).sum().item()
            tokens   += (b["labels"] != -100).sum().item()
    return math.exp(loss_sum / tokens)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="viro_dummy_data/tokenized/train")
    ap.add_argument("--valid_dir", default="viro_dummy_data/tokenized/validation")
    ap.add_argument("--tokenizer",  default="tokenizer")
    ap.add_argument("--ckpt_dir",   default="ckpts")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float,  default=1e-4)
    args = ap.parse_args()

    accel = Accelerator()
    train_dl, tok = get_dataloader(args.train_dir, args.tokenizer, args.batch, True)
    val_dl, _     = get_dataloader(args.valid_dir, args.tokenizer, args.batch, False)

    model = GPTSmall(len(tok), tok.pad_token_id)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = CosineAnnealingLR(opt, args.epochs)

    model, opt, train_dl, val_dl = accel.prepare(model, opt, train_dl, val_dl)
    Path(args.ckpt_dir).mkdir(exist_ok=True)

    best = float("inf"); patience=0
    for ep in range(1, args.epochs+1):
        model.train(); start=time.time()
        for batch in track(train_dl, description=f"[bold blue]Ep {ep}"):
            opt.zero_grad()
            loss = cross_entropy(
                model(batch["input_ids"]).view(-1, len(tok)),
                batch["labels"].view(-1),
                ignore_index=-100)
            accel.backward(loss); opt.step()
        sched.step()
        ppl = evaluate(model, val_dl, tok, accel)
        accel.print(f"Epoch {ep} | val PPL {ppl:.2f} | {time.time()-start:.1f}s")
        if accel.is_main_process:
            accel.save_state(args.ckpt_dir)
        if ppl+0.01 < best: best, patience= ppl,0
        else:
            patience += 1
            if patience >= 3: accel.print("Early stop."); break

if __name__ == "__main__":
    main()
