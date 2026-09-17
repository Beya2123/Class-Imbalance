"""Torch Dataset for transformer models (M3, M4, M5)."""
import torch
from torch.utils.data import Dataset
from typing import List, Optional

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]],
                 tokenizer, max_len: int = 128):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
        )
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}
