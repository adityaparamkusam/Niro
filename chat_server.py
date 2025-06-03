import torch, sentencepiece as spm, uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from model import GPTSmall

tok=spm.SentencePieceProcessor(model_file="niro_bpe.model")
model=GPTSmall(tok.vocab_size(),tok.pad_id())
model.load_state_dict(torch.load("checkpoints/niro_final.pt",map_location="cpu"))
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

class Req(BaseModel): prompt:str; max_new:int=128
app=FastAPI()

@app.post("/generate")
def gen(r:Req):
    ids=torch.tensor([tok.encode(r.prompt,out_type=int)]).to(next(model.parameters()).device)
    out=model.generate(ids,max_new=r.max_new,eos=tok.eos_id())
    return {"text": tok.decode(out[0].cpu().tolist())}

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000,workers=1)