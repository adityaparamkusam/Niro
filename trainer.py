import torch, torch.nn.functional as F, argparse, os, math, random
from accelerate import Accelerator
from tqdm import tqdm
from model import GPTSmall
from data_utils import StreamDataset
from optimizer import make_optimizer
import sentencepiece as spm

ap=argparse.ArgumentParser()
ap.add_argument("--tok", default="niro_bpe.model")
ap.add_argument("--data_glob", default="Data/*.bin")
ap.add_argument("--seq_len", type=int, default=1024)
ap.add_argument("--steps", type=int, default=400_000)
ap.add_argument("--batch_tokens", type=int, default=1_048_576)  # ≈ 1024 * 1024
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--warmup", type=int, default=5_000)
ap.add_argument("--weight_decay", type=float, default=0.1)
ap.add_argument("--save_dir", default="checkpoints")
opt=ap.parse_args()

tok=spm.SentencePieceProcessor(model_file=opt.tok)
model=GPTSmall(tok.vocab_size(), tok.pad_id(), checkpoint=True)
model=torch.compile(model)  # PyTorch 2.3

ds=StreamDataset(opt.data_glob,opt.seq_len)
dl=torch.utils.data.DataLoader(ds,batch_size=opt.batch_tokens//opt.seq_len,
                               num_workers=4, pin_memory=True)

accelerator=Accelerator(mixed_precision="bf16")
model,dl=accelerator.prepare(model,dl)

optimizer=make_optimizer(model.parameters(), opt.lr, opt.weight_decay,
                         opt.warmup, opt.steps)

os.makedirs(opt.save_dir,exist_ok=True)
prog=tqdm(total=opt.steps, disable=not accelerator.is_main_process)
loss_avg=0; step=0

for batch in dl:
    step+=1
    logits=model(batch[:,:-1])
    loss=F.cross_entropy(logits.reshape(-1, tok.vocab_size()),
                         batch[:,1:].reshape(-1),
                         ignore_index=tok.pad_id())
    loss=loss/opt.accum; accelerator.backward(loss)
    if step%opt.accum==0:
        optimizer.step(); optimizer.zero_grad()
    loss_avg+=loss.item()*opt.accum; prog.update(1)
    if step%200==0 and accelerator.is_main_process:
        prog.set_description(f"step {step} loss {loss_avg/200:.3f}")
        loss_avg=0
    if step%10_000==0 and accelerator.is_main_process:
        ck=f"{opt.save_dir}/niro_{step}.pt"; torch.save(model.state_dict(),ck); print(f"💾 {ck}")
    if step>=opt.steps: break

if accelerator.is_main_process:
    final=f"{opt.save_dir}/niro_final.pt"
    torch.save(model.state_dict(),final); print(f"✅ training done → {final}")