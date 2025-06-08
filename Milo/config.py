from dataclasses import dataclass, asdict
import json, math, os

@dataclass
class ModelConfig:
    vocab_size   : int = 50_257       # must match tokenizer
    block_size   : int = 1024         # context length
    n_layer      : int = 36           # Increased from 32
    n_head       : int = 20
    n_embd       : int = 1280         # Option to increase to 1408
    dropout      : float = 0.05       # Reduced from 0.1

@dataclass
class TrainConfig:
    device        : str   = "mps"     # 'cuda' on EC2, 'mps' on Apple
    batch_tokens  : int   = 2048      # context × batch_size (fits 18 GB unified mem)
    grad_accum    : int   = 8         # effective tokens/step = batch_tokens × grad_accum
    lr            : float = 3e-4      # Adjusted from 4e-4
    warmup_steps  : int   = 2500      # Adjusted from 2000
    max_steps     : int   = 50_000    # raise after basic sanity
    log_every     : int   = 100
    ckpt_every    : int   = 1000
    model_dir     : str   = "checkpoints"

# handy helpers
def save_config(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from dataclasses import is_dataclass, asdict
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_config(path):
    with open(path) as f: d = json.load(f)
    m_cfg = ModelConfig(**{k:v for k,v in d.items() if k in ModelConfig.__annotations__})
    t_cfg = TrainConfig(**{k:v for k,v in d.items() if k in TrainConfig.__annotations__})
    return m_cfg, t_cfg