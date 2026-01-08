"""
Data loading utilities for Attention Is Not All You Need experiments.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
from typing import Optional

# SNLI dataset
from .snli import SNLIDataset

# WikiText dataset
from .wikitext import Wikitext2Dataset


class TextDataset(Dataset):
    """Dataset for text data."""

    def __init__(self, texts: list, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for text in texts:
            # Skip empty texts
            if not text.strip():
                continue

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)

            # Create sequences of max_length
            for i in range(0, len(tokens) - max_length, max_length // 2):
                seq = tokens[i:i + max_length + 1]  # +1 for target
                if len(seq) == max_length + 1:
                    self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def get_tokenizer(model_name: str = "gpt2"):
    """Get tokenizer for the specified model."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_wikitext(split: str = "train", max_samples: Optional[int] = None):
    """Load Wikitext-2 dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    texts = [item["text"] for item in dataset]
    if max_samples:
        texts = texts[:max_samples]

    return texts


def get_dataloader(texts: list, tokenizer=None, batch_size: int = 32,
                  max_length: int = 256, shuffle: bool = True, num_workers: int = 4):
    """Create DataLoader from text data."""
    if tokenizer is None:
        tokenizer = get_tokenizer()

    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader