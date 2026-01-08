"""SNLI dataset for natural language inference classification.

This module provides the SNLIDataset class for loading and preprocessing
the SNLI (Stanford Natural Language Inference) dataset with paired sentence
encoding suitable for DistilBERT-based models.
"""

import torch
from datasets import load_dataset


class SNLIDataset(torch.utils.data.Dataset):
    """SNLI dataset for classification.
    
    Loads the SNLI dataset and handles paired tokenization of premise-hypothesis
    pairs using DistilBERT's sentence-pair encoding strategy. This allows the
    model to learn cross-sentence attention patterns essential for NLI tasks.
    
    The tokenizer encodes both sentences together as:
        [CLS] premise [SEP] hypothesis [SEP]
    
    This enables:
    - Joint attention across both sentences
    - Proper [SEP] token placement
    - Correct position embeddings
    - [CLS] token aggregates pair-level information
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        tokenizer: HuggingFace tokenizer (e.g., DistilBertTokenizer)
        max_seq_len: Maximum sequence length per sentence (default: 48)
                     Total max length will be max_seq_len * 2 for both sentences
    """
    
    def __init__(self, split: str, tokenizer, max_seq_len: int = 48):
        """Initialize SNLI dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            tokenizer: Tokenizer for encoding text
            max_seq_len: Maximum sequence length per sentence
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Load SNLI
        dataset = load_dataset("snli", split=split)
        
        # Filter out examples with label -1 (no gold label)
        self.examples = []
        for example in dataset:
            if example["label"] != -1:
                self.examples.append({
                    "premise": example["premise"],
                    "hypothesis": example["hypothesis"],
                    "label": example["label"]
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize premise and hypothesis together (DistilBERT paired encoding)
        # This creates: [CLS] premise [SEP] hypothesis [SEP]
        # Allows cross-attention between sentences - crucial for NLI
        encoding = self.tokenizer(
            example["premise"],
            example["hypothesis"],
            max_length=self.max_seq_len * 2,  # Both sentences
            padding="max_length",
            truncation='only_first',  # Truncate only premise if needed (preserves full hypothesis)
            return_overflowing_tokens=False,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(example["label"], dtype=torch.long)
        }
