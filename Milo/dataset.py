import random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

class ArrowStreamDataset(Dataset):
    def __init__(self, paths, block_size: int):
        self.datasets   = [load_from_disk(p) for p in paths]
        self.block_size = block_size

        # estimate total tokens for __len__ and proportional sampling
        def row_len(col):            # works for list[int] rows
            return sum(len(r) for r in col)
        self.token_lens  = [row_len(ds["input_ids"]) for ds in self.datasets]
        self.cum_tokens  = np.cumsum(self.token_lens)
        self.total_tokens = int(self.cum_tokens[-1])

    def __len__(self):
        return self.total_tokens // self.block_size


    def _sample_row(self):
        pick  = random.randint(0, self.total_tokens - 1)
        shard = int(np.searchsorted(self.cum_tokens, pick))
        return random.choice(self.datasets[shard]["input_ids"])

    def __getitem__(self, idx):
        row = self._sample_row()
        while len(row) < self.block_size + 1:          # pad very short rows
            row = row + row
        start  = random.randint(0, len(row) - self.block_size - 1)
        chunk  = row[start : start + self.block_size + 1]
        x      = torch.tensor(chunk[:-1], dtype=torch.long)
        y      = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


def create_loader(paths, m_cfg, t_cfg):
    ds = ArrowStreamDataset(paths, m_cfg.block_size)
    bs = max(1, t_cfg.batch_tokens // m_cfg.block_size)
    return DataLoader(
        ds, batch_size=bs, shuffle=True,
        num_workers=2, pin_memory=True,
        prefetch_factor=4, persistent_workers=True
    )