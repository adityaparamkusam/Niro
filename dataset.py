from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from rich import print as rprint

def get_dataloader(
    split_dir: str | Path,
    tokenizer_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    workers: int = 4,
) -> Tuple[DataLoader, "AutoTokenizer"]:
    split_dir = Path(split_dir).expanduser().resolve()
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)
    ds = load_from_disk(str(split_dir))
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    coll = DataCollatorForLanguageModeling(tok, mlm=False)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=coll,
        num_workers=max(workers, 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    rprint(f"[green]Loaded {split_dir.name}: {len(ds):,} samples[/green]")
    return dl, tok
