import torch
import numpy as np
from typing import List, Dict, Any

def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Computes group relative advantages.
    
    Args:
        rewards: Tensor of shape (batch_size, group_size) containing rewards for each rollout.
        eps: Small constant for numerical stability.
        
    Returns:
        advantages: Tensor of shape (batch_size, group_size) containing normalized advantages.
    """
    # rewards shape: [B, G]
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    
    # Advantage = (r - mean) / (std + eps)
    advantages = (rewards - mean) / (std + eps)
    return advantages

def log_training_metrics(metrics: Dict[str, Any], step: int):
    """
    Helper to log metrics to console or wandb if configured.
    This is a placeholder for more complex logging logic if needed.
    """
    # In a real implementation, this might interface with wandb directly
    # or just format strings for the logger.
    log_str = f"Step {step}: "
    for k, v in metrics.items():
        if isinstance(v, float):
            log_str += f"{k}={v:.4f} "
        else:
            log_str += f"{k}={v} "
    # We assume standard logging is handled by the Trainer, this is for custom debug
    pass
