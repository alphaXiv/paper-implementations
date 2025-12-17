"""Training utilities and main training loop."""

import os
import math
from typing import Any, cast, List
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from tiny_recursive_models.training.config import PretrainConfig, TrainState
from tiny_recursive_models.training.checkpoint import load_checkpoint
from tiny_recursive_models.utils import load_model_class

# Import optimizer from our vendored implementation
try:
    from tiny_recursive_models.training.optimizers import AdamATan2
except Exception:
    AdamATan2 = None

# Import from trm package
from tiny_recursive_models.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from tiny_recursive_models.data.common import PuzzleDatasetMetadata
from tiny_recursive_models.models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    """Create a DataLoader for training or evaluation.
    
    Args:
        config: Training configuration
        split: 'train' or 'test'
        rank: Current process rank
        world_size: Total number of processes
        **kwargs: Additional arguments for PuzzleDatasetConfig
        
    Returns:
        Tuple of (dataloader, metadata)
    """
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test) > 0 and split == "test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, is_eval: bool = False):
    """Create and initialize model with loss head and optimizers.
    
    Args:
        config: Training configuration
        train_metadata: Dataset metadata
        rank: Current process rank
        world_size: Total number of processes
        is_eval: If True, skip optimizer creation
        
    Returns:
        Tuple of (model, optimizers, optimizer_lrs)
    """
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        # Default to NOT compiling unless explicitly opted-in. This avoids
        # torch.compile wrapping that introduces the "_orig_mod." prefix in
        # state_dict keys and causes checkpoint mismatches at load time.
        if os.environ.get("ENABLE_COMPILE", "0") == "1":
            try:
                model = torch.compile(model)  # type: ignore
            except Exception as _e:
                print("torch.compile failed; proceeding without compilation:", _e)

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    # For evaluation-only usage we skip creating optimizers to avoid requiring
    # optimizer packages that may not be installed in an eval environment.
    if is_eval:
        return model, [], []

    # If AdamATan2 isn't available, fail early with a helpful message when
    # an optimizer that requires it would be created. This avoids confusing
    # "None is not callable" errors later and gives a clear remediation.
    need_adam = not config.freeze_weights and getattr(config.arch, "puzzle_emb_ndim", None) != 0
    if need_adam and AdamATan2 is None:
        raise RuntimeError(
            "adam_atan2 package is required for training optimizers but was not found. "
            "Install it (pip install <package>) or run in evaluation mode by passing is_eval=True."
        )
    if getattr(config.arch, 'puzzle_emb_ndim', 0) == 0:
        optimizers = [
            cast(Any, AdamATan2)(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            cast(Any, AdamATan2)(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    """Cosine learning rate schedule with warmup.
    
    Args:
        current_step: Current training step
        base_lr: Base learning rate
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_ratio: Minimum LR as fraction of base_lr
        num_cycles: Number of cosine cycles
        
    Returns:
        Current learning rate
    """
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    """Compute current learning rate based on schedule.
    
    Args:
        base_lr: Base learning rate
        config: Training configuration
        train_state: Current training state
        
    Returns:
        Current learning rate
    """
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, is_eval: bool = False):
    """Initialize training state with model and optimizers.
    
    Args:
        config: Training configuration
        train_metadata: Dataset metadata
        rank: Current process rank
        world_size: Total number of processes
        is_eval: If True, skip optimizer creation
        
    Returns:
        TrainState object
    """
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size, is_eval=is_eval)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    """Train on a single batch.
    
    Args:
        config: Training configuration
        train_state: Current training state
        batch: Input batch
        global_batch_size: Global batch size across all processes
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        Dictionary of metrics (on rank 0 only)
    """
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def mix_weights_direct(device, alpha, net, nets):
    """Mix weights from multiple networks (utility function).
    
    Args:
        device: Device to perform mixing on
        alpha: Mixing coefficients
        net: Target network
        nets: List of source networks
        
    Returns:
        Network with mixed weights
    """
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0] * sd[0][k].to(device)
        for i in range(1, len(nets)):
            comb_net += alpha[i] * sd[i][k].to(device)
        sd_alpha[k] = comb_net
    net.load_state_dict(sd_alpha)
    return net
