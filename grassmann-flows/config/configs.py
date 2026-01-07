"""
Predefined configurations for different experiment scales.

Usage:
    from configs import CONFIGS
    config = CONFIGS["small"]
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    model_dim: int
    num_layers: int
    num_heads: int  # For GPT-2
    reduced_dim: int  # For Grassmann
    ff_dim: int
    max_seq_len: int
    window_sizes: List[int]
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    epochs: int


# Model configurations at different scales
MODEL_CONFIGS = {
    # Tiny model for debugging (~2M params)
    "tiny": ModelConfig(
        name="tiny",
        model_dim=128,
        num_layers=4,
        num_heads=4,
        reduced_dim=32,
        ff_dim=512,
        max_seq_len=128,
        window_sizes=[1, 2, 4, 8],
        dropout=0.1,
    ),

    # Small model for quick experiments (~15M params)
    "small": ModelConfig(
        name="small",
        model_dim=256,
        num_layers=6,
        num_heads=4,
        reduced_dim=32,
        ff_dim=1024,
        max_seq_len=256,
        window_sizes=[1, 2, 4, 8, 16],
        dropout=0.1,
    ),

    # Medium model (~50M params)
    "medium": ModelConfig(
        name="medium",
        model_dim=512,
        num_layers=8,
        num_heads=8,
        reduced_dim=64,
        ff_dim=2048,
        max_seq_len=512,
        window_sizes=[1, 2, 4, 8, 16, 32, 64],
        dropout=0.1,
    ),

    # GPT-2 scale (~124M params)
    "gpt2": ModelConfig(
        name="gpt2",
        model_dim=768,
        num_layers=12,
        num_heads=12,
        reduced_dim=96,
        ff_dim=3072,
        max_seq_len=1024,
        window_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
        dropout=0.1,
    ),

    # Large model (~350M params)
    "large": ModelConfig(
        name="large",
        model_dim=1024,
        num_layers=24,
        num_heads=16,
        reduced_dim=128,
        ff_dim=4096,
        max_seq_len=1024,
        window_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        dropout=0.1,
    ),
}


# Training configurations
TRAIN_CONFIGS = {
    # Fast training for debugging
    "debug": TrainConfig(
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=100,
        max_grad_norm=1.0,
        epochs=1,
    ),

    # Quick training for experiments
    "quick": TrainConfig(
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=500,
        max_grad_norm=1.0,
        epochs=5,
    ),

    # Standard training
    "standard": TrainConfig(
        batch_size=64,
        learning_rate=6e-4,
        weight_decay=0.1,
        warmup_steps=2000,
        max_grad_norm=1.0,
        epochs=10,
    ),

    # Full pretraining
    "full": TrainConfig(
        batch_size=128,
        learning_rate=6e-4,
        weight_decay=0.1,
        warmup_steps=5000,
        max_grad_norm=1.0,
        epochs=20,
    ),
}


def get_config(model_scale: str = "small", train_mode: str = "quick"):
    """Get combined model and training config."""
    if model_scale not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model scale: {model_scale}. Choose from {list(MODEL_CONFIGS.keys())}")
    if train_mode not in TRAIN_CONFIGS:
        raise ValueError(f"Unknown train mode: {train_mode}. Choose from {list(TRAIN_CONFIGS.keys())}")

    return MODEL_CONFIGS[model_scale], TRAIN_CONFIGS[train_mode]


def estimate_params(config: ModelConfig, vocab_size: int = 50257) -> dict:
    """Estimate parameter counts for both model types."""
    d = config.model_dim
    n = config.num_layers
    h = config.num_heads
    r = config.reduced_dim
    ff = config.ff_dim
    V = vocab_size

    # Shared: embeddings
    embed_params = V * d + config.max_seq_len * d

    # GPT-2 per layer: attention (4*d*d) + FFN (2*d*ff) + 2*LayerNorm (4*d)
    gpt2_layer = 4 * d * d + 2 * d * ff + 4 * d
    gpt2_total = embed_params + n * gpt2_layer + 2 * d  # +final LN

    # Grassmann per layer:
    # - Reduction: d*r + r
    # - Plucker proj: (r*(r-1)/2)*d + d
    # - Gate: 2d*d + d
    # - FFN: 2*d*ff + ff + d
    # - 2*LayerNorm: 4*d
    plucker_dim = r * (r - 1) // 2
    grassmann_layer = (d * r + r) + (plucker_dim * d + d) + (2 * d * d + d) + (2 * d * ff + ff + d) + 4 * d
    grassmann_total = embed_params + n * grassmann_layer + 2 * d

    return {
        "gpt2": gpt2_total,
        "grassmann": grassmann_total,
        "gpt2_M": gpt2_total / 1e6,
        "grassmann_M": grassmann_total / 1e6,
    }


if __name__ == "__main__":
    # Print parameter estimates for all configs
    print("Parameter estimates (approximate):\n")
    print(f"{'Config':<10} {'GPT-2 (M)':<12} {'Grassmann (M)':<15}")
    print("-" * 40)

    for name, config in MODEL_CONFIGS.items():
        params = estimate_params(config)
        print(f"{name:<10} {params['gpt2_M']:<12.1f} {params['grassmann_M']:<15.1f}")
