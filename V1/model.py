from __future__ import annotations
import torch, torch.nn as nn

class CausalSelfAttn(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_p: float):
        super().__init__()
        self.nh, self.dk = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(attn_p)
        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.nh, self.dk).transpose(1, 2)            # [B,h,T,d]
        k = k.view(B, T, self.nh, self.dk).transpose(1, 2)
        v = v.view(B, T, self.nh, self.dk).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / self.dk**0.5
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, -1e4)
        att = self.drop(att.softmax(dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(p),
            nn.Linear(d_ff, d_model), nn.Dropout(p)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, p: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttn(d_model, n_head, p)
        self.mlp  = MLP(d_model, d_ff, p)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTSmall(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        n_layer=6,
        n_head=8,
        d_model=512,
        d_ff=2048,
        max_seq=1024,
        p=0.1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq, d_model))
        self.blocks  = nn.Sequential(*[Block(d_model, n_head, d_ff, p) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("tokenizer")
    print(f"Params: {GPTSmall(len(t), t.pad_token_id).parameters().__sizeof__()}")
