import math, torch, torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head   = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv      = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj     = nn.Linear(cfg.n_embd,     cfg.n_embd, bias=False)
        self.drop     = nn.Dropout(cfg.dropout)

    # flash-attention path
    def forward(self, x):
        B, T, C = x.size()
        qkv = (
            self.qkv(x)
            .view(B, T, 3, self.n_head, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv                                    # each (B,h,T,hd)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp (self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb   = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.n_embd)
        self.head      = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight      # weight-tying
        self.use_checkpoint = True
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = cp.checkpoint(block, x) if self.use_checkpoint else block(x)

        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)