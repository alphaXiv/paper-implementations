"""WikiText-2 dataset for language modeling.

This module provides the Wikitext2Dataset class for loading and preprocessing
the WikiText-2 dataset for causal language modeling tasks.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class Wikitext2Dataset(Dataset):
    """Wikitext-2 dataset for language modeling.
    
    Loads the WikiText-2 dataset and creates fixed-length chunks for
    causal language modeling. The dataset concatenates all text and
    splits it into non-overlapping sequences of max_seq_len tokens.
    
    For language modeling, both input and target are the same sequence,
    where the model predicts the next token at each position.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        tokenizer: HuggingFace tokenizer (e.g., BertTokenizer, GPT2Tokenizer)
        max_seq_len: Maximum sequence length (default: 256)
    """
    
    def __init__(self, split: str, tokenizer, max_seq_len: int = 256):
        """Initialize the dataset.

        Args:
            split (str): Dataset split ('train', 'validation', 'test').
            tokenizer: Tokenizer for encoding text.
            max_seq_len (int): Maximum sequence length.
        """
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        # Load wikitext-2
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Concatenate all text and tokenize
        # Note: We encode the full text (2M+ tokens) then chunk it ourselves
        # Suppress tokenizer's max_length warning since we handle chunking manually
        all_text = "\n".join([t for t in dataset["text"] if t.strip()])
        self.tokens = tokenizer.encode(all_text, verbose=False)

        # Create chunks
        self.num_chunks = len(self.tokens) // max_seq_len
        self.tokens = self.tokens[:self.num_chunks * max_seq_len]

    def __len__(self):
        """Return the number of chunks in the dataset.

        Returns:
            int: Number of chunks.
        """
        return self.num_chunks

    def __getitem__(self, idx):
        """Get a chunk of tokens.

        Args:
            idx (int): Index of the chunk.

        Returns:
            tuple: (input_ids, target_ids) both of shape (max_seq_len,).
                   For language modeling, input and target are identical.
        """
        start = idx * self.max_seq_len
        chunk = self.tokens[start:start + self.max_seq_len]
        x = torch.tensor(chunk, dtype=torch.long)
        return x, x.clone()  # input and target are same for LM
