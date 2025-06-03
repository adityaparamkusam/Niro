from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

try: from flash_attn import flash_attn_func
except ImportError: flash_attn_func = None   # graceful fallback

# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    vocab_size:int; pad_id:int
    d_model:int=768; n_head:int=12; n_layer:int=14
    d_ff:int=3072; max_seq:int=2048
    dropout:float=0.1; attn_dropout:float=0.0
    rope_pct:float=1.0; checkpoint:bool=False

# ───── Layer helpers ───────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self,d,eps=1e-5): super().__init__(); self.w=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x): return self.w * x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)

class RoPE(nn.Module):                          # rotary embedding
    def __init__(self,dim,max_seq=2048,base=10_000):
        super().__init__(); t=torch.arange(max_seq); inv=1.0/(base**(torch.arange(0,dim,2)/dim))
        freqs=torch.einsum('i,j->ij',t.float(),inv)
        self.register_buffer("cos",freqs.cos(),False); self.register_buffer("sin",freqs.sin(),False); self.dim=dim
    def forward(self,q,k,pos):
        cos,sin=self.cos[pos],self.sin[pos]
        def apply(x):
            x1,x2=x[...,::2],x[...,1::2]
            return torch.cat([x1*cos-x2*sin,x1*sin+x2*cos],-1)
        return apply(q),apply(k)

class MHA(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__(); self.h=cfg.n_head; self.dk=cfg.d_model//cfg.n_head; self.cfg=cfg
        self.qkv=nn.Linear(cfg.d_model,3*cfg.d_model,bias=False)
        self.proj=nn.Linear(cfg.d_model,cfg.d_model,bias=False)
        rope_dim=int(self.dk*cfg.rope_pct); self.rope=RoPE(rope_dim,cfg.max_seq) if rope_dim else None
        self.scale=0.81*math.sqrt(cfg.n_layer) if cfg.n_layer>12 else 1.0
    def _split(self,x): B,T,_=x.shape; return x.view(B,T,self.h,self.dk).transpose(1,2)
    def _merge(self,x): return x.transpose(1,2).contiguous().view(x.size(0),-1,self.h*self.dk)
    def forward(self,x):
        q,k,v=self.qkv(x).chunk(3,-1); q,k,v=map(self._split,(q,k,v))
        if self.rope: q,k=self.rope(q,k,pos=torch.arange(x.size(1),device=x.device))
        if flash_attn_func:
            q,k,v=[t.contiguous().to(torch.float16) for t in (q,k,v)]
            attn=flash_attn_func(q,k,v,dropout_p=self.cfg.attn_dropout,causal=True).to(x.dtype)
        else:
            attn=F.scaled_dot_product_attention(q,k,v,dropout_p=self.cfg.attn_dropout,is_causal=True)
        return self.proj(self._merge(attn))*self.scale

class SwiGLU(nn.Module):
    def __init__(self,cfg): super().__init__(); self.fc1=nn.Linear(cfg.d_model,2*cfg.d_ff,bias=False); self.fc2=nn.Linear(cfg.d_ff,cfg.d_model,bias=False)
    def forward(self,x): x,g=self.fc1(x).chunk(2,-1); return self.fc2(F.silu(g)*x)

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.ln1=RMSNorm(cfg.d_model); self.ln2=RMSNorm(cfg.d_model)
        self.attn=MHA(cfg); self.mlp=SwiGLU(cfg); self.drop=nn.Dropout(cfg.dropout)
    def forward(self,x):
        x=x+self.drop(self.attn(self.ln1(x))); x=x+self.drop(self.mlp(self.ln2(x))); return x

# ───── Full network ────────────────────────────────────────────────────────────
class GPTSmall(nn.Module):
    def __init__(self,vocab_size:int,pad_id:int,**kw):
        super().__init__(); self.cfg=Config(vocab_size,pad_id,**kw)
        c=self.cfg
        self.tok_emb=nn.Embedding(c.vocab_size,c.d_model,padding_idx=pad_id)
        self.drop=nn.Dropout(c.dropout)
        self.blocks=nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f=RMSNorm(c.d_model)
        self.head=nn.Linear(c.d_model,c.vocab_size,bias=False); self.head.weight=self.tok_emb.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,0.02); nn.init.zeros_(m.bias) if m.bias is not None else None
        if isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def forward(self,idx):
        x=self.tok_emb(idx); x=self.drop(x)
        for blk in self.blocks: x=blk(x)
        return self.head(self.ln_f(x))
    @torch.inference_mode()
    def generate(self,idx,max_new=128,top_p=0.9,temp=1.0,eos=None):
        device=idx.device
        for _ in range(max_new):
            logits=self(idx)[:,-1,:]/temp; probs=F.softmax(logits,-1)
            sorted_p,sorted_i=torch.sort(probs,desc=True); cumsum=sorted_p.cumsum(-1)
            keep= cumsum<=top_p; sorted_p[~keep]=0; sorted_p/=sorted_p.sum(-1,keepdim=True)
            next_id=sorted_i.gather(-1,torch.multinomial(sorted_p,1))
            idx=torch.cat([idx,next_id],1)
            if eos is not None and next_id.item()==eos: break
        return idx