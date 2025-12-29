"""
Tensor helper for tool calling interactions
"""

import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class TensorConfig:
    pad_token_id: int


class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor], 
                             keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:
        """Cut tensors to their effective length based on attention mask."""
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()
        
        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> torch.Tensor:
        """Convert padding structure by sorting and removing padding tokens.
        
        Args:
            tensor: Input tensor
            pad_to_left: If True, move content to right and padding to left
                        If False, move content to left and padding to right
        
        Returns:
            Tensor with tokens sorted and padding removed
        """
        # Create mask for content vs padding
        # If pad_to_left=True: content=1, padding=0 to move content right
        # If pad_to_left=False: content=0, padding=1 to move content left
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        
        # Sort to move content to desired side
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        sorted_tensor = tensor.gather(1, sorted_indices)
        
        # Calculate effective length (non-padding tokens)
        effective_len = (tensor != self.config.pad_token_id).sum(dim=1).max().item()
        
        # Keep only the content side
        if pad_to_left:
            return sorted_tensor[:, -effective_len:]
        else:
            return sorted_tensor[:, :effective_len]

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids."""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position ids from attention mask."""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: List[torch.Tensor], 
                                pad_to_left: bool = True) -> torch.Tensor:
        """Concatenate tensors and handle padding."""
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor