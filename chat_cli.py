import torch,sentencepiece as spm,readline,os
from model import GPTSmall

tok=spm.SentencePieceProcessor(model_file="niro_bpe.model")
model=GPTSmall(tok.vocab_size(),tok.pad_id())
model.load_state_dict(torch.load("checkpoints/niro_final.pt",map_location="cpu"))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()

ctx=""
while True:
    try: usr=input("You: "); ctx+=f"\nUser: {usr}\nAssistant:"
    except EOFError: break
    ids=torch.tensor([tok.encode(ctx.strip(),out_type=int)]).to(device)
    out=model.generate(ids,max_new=128,eos=tok.eos_id())
    reply=tok.decode(out[0].cpu().tolist()).split("Assistant:")[-1].strip()
    ctx+=f" {reply}"
    print("Niro:",reply)