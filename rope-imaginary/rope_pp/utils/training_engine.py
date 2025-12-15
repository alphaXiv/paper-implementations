"""
Custom training engine that replaces HuggingFace Trainer.
Supports both single-GPU and multi-GPU (DeepSpeed) training using Accelerate.
"""

import os
import time
import torch
import wandb
from datetime import datetime
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import DefaultDataCollator

from utils.dataset_utils import (
    StreamingTrainingParquet, 
    StreamingTrainingJsonlZSD, 
    StreamingTrainingHuggingFace
)


def get_rank_and_world_size():
    """Get distributed training info, works for both distributed and single GPU."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def format_time(seconds):
    """Format seconds into human-readable time (e.g., 1.23m, 45.67s, 2.34h)"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    else:
        return f"{seconds / 3600:.2f}h"


def save_checkpoint(
    accelerator: Accelerator,
    train_dataset,
    step: int,
    output_dir: str,
):
    """
    Save model, optimizer, scheduler, and dataset state using Accelerate.
    
    Args:
        accelerator: Accelerator instance
        train_dataset: The training dataset (for dataset checkpointing)
        step: Current training step
        output_dir: Directory to save checkpoint
    """
    rank, world_size = get_rank_and_world_size()
    
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    
    # Accelerate handles model/optimizer/scheduler saving
    accelerator.save_state(checkpoint_dir)
    
    # Also save step number
    if rank == 0:
        step_file = os.path.join(checkpoint_dir, 'step.txt')
        with open(step_file, 'w') as f:
            f.write(str(step))
    
    # Save dataset state (works for both DeepSpeed and standard)
    dataset_ckpt_path = os.path.join(
        checkpoint_dir, 
        f"dataset_ckpt-{rank:0{len(str(world_size))}d}-{world_size}.pt"
    )
    
    if isinstance(train_dataset, StreamingTrainingParquet):
        dataset_ckpt = {
            'data_path': train_dataset.data_path,
            'label_name': train_dataset.label_name,
            'pivot': train_dataset.pivot,
            'size': train_dataset.size,
            'table_idx': train_dataset.table_idx,
            'table_num': train_dataset.table_num,
            'table_buffer': train_dataset.table_buffer,
            'sample_idx': train_dataset.sample_idx,
            'sample_num': train_dataset.sample_num,
            'token_buffer': train_dataset.token_buffer,
        }
        torch.save(dataset_ckpt, dataset_ckpt_path)
        
    elif isinstance(train_dataset, StreamingTrainingJsonlZSD):
        dataset_ckpt = {
            'data_path': train_dataset.data_path,
            'label_name': train_dataset.label_name,
            'pivot': train_dataset.pivot,
            'size': train_dataset.size,
            'sample_idx': train_dataset.sample_idx,
            'token_buffer': train_dataset.token_buffer,
        }
        torch.save(dataset_ckpt, dataset_ckpt_path)
        
    elif isinstance(train_dataset, StreamingTrainingHuggingFace):
        dataset_ckpt = {
            'token_buffer': train_dataset.token_buffer,
        }
        torch.save(dataset_ckpt, dataset_ckpt_path)
    
    if rank == 0:
        print(f"Checkpoint saved at step {step} to {checkpoint_dir}")


def load_checkpoint(
    accelerator: Accelerator,
    checkpoint_path: str,
) -> int:
    """
    Load checkpoint using Accelerate.
    
    Returns:
        The step number from the checkpoint
    """
    rank, _ = get_rank_and_world_size()
    
    # Load the accelerator state
    accelerator.load_state(checkpoint_path)
    
    # Load step number
    step_file = os.path.join(checkpoint_path, 'step.txt')
    step = 0
    if os.path.exists(step_file):
        with open(step_file, 'r') as f:
            step = int(f.read().strip())
    
    if rank == 0:
        print(f"Loaded checkpoint from {checkpoint_path}, resuming from step {step}")
    
    return step

@torch.no_grad()
def evaluate(
    model,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
) -> Dict[str, float]:
    """
    Run evaluation on the validation dataset.
    
    Returns:
        Dictionary with evaluation metrics (loss)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    eval_start = time.time()
    
    for batch in eval_dataloader:
        # Batch is already on the right device thanks to Accelerate
        outputs = model(**batch)
        loss = outputs.loss
        
        total_loss += loss.item()
        num_batches += 1
    
    # Average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Gather losses from all ranks if distributed
    avg_loss = accelerator.gather(torch.tensor([avg_loss], device=accelerator.device)).mean().item()
    
    eval_time = time.time() - eval_start
    
    model.train()
    
    return {
        'loss': avg_loss,
        'runtime': eval_time,
    }


def log_training_step(
    step: int,
    max_steps: int,
    loss: float,
    grad_norm: float,
    learning_rate: float,
    base_lr: float,
    step_time_ms: float,
    batch_token: int,
    world_size: int,
    start_time: datetime,
    valid_total_time: float,
    logging_steps: int = 1,
):
    """Log training metrics in nanochat style."""
    rank, _ = get_rank_and_world_size()
    
    if rank != 0 or step % logging_steps != 0:
        return
    
    # Calculate time metrics
    total_time = (datetime.now() - start_time).total_seconds()
    train_total_time = total_time - valid_total_time
    
    # Calculate token metrics
    num_consume_token = step * batch_token
    tokens_per_sec = (batch_token / (step_time_ms / 1000)) if step_time_ms > 0 else 0
    avg_tgs = num_consume_token / train_total_time / world_size if train_total_time > 0 else 0
    
    # Calculate progress
    cur_percent = step / max_steps * 100
    
    # Calculate learning rate multiplier
    lr_multiplier = learning_rate / base_lr if base_lr > 0 else 1.0
    
    # Format step string with leading zeros
    step_str = f"{step:0{len(str(max_steps))}d}"
    
    # Nanochat-style log format
    log_str = (
        f"step {step_str}/{max_steps} ({cur_percent:.2f}%) | "
        f"loss: {loss:.6f} | "
        f"grad norm: {grad_norm:.4f} | "
        f"lr: {learning_rate:.2e} | "
        f"lrm: {lr_multiplier:.2f} | "
        f"dt: {step_time_ms:.2f}ms | "
        f"tok/sec: {tokens_per_sec:,.0f} | "
        f"avg tok/sec: {avg_tgs:,.0f} | "
        f"total time: {format_time(total_time)}"
    )
    print(log_str)
    
    # Log to WandB
    wandb.log({
        'train/loss': loss,
        'train/grad_norm': grad_norm,
        'train/learning_rate': learning_rate,
        'train/tokens_per_sec': tokens_per_sec,
        'train/avg_tokens_per_sec': avg_tgs,
        'train/step_time_ms': step_time_ms,
        'step': step,
    })


def log_evaluation(
    step: int,
    max_steps: int,
    eval_metrics: Dict[str, float],
    dataset_abbr: str,
    start_time: datetime,
):
    """Log evaluation metrics."""
    rank, _ = get_rank_and_world_size()
    
    if rank != 0:
        return
    
    total_time = (datetime.now() - start_time).total_seconds()
    cur_percent = step / max_steps * 100
    
    step_str = f"{step:0{len(str(max_steps))}d}"
    eval_loss = eval_metrics.get('loss', 0.0)
    
    eval_str = (
        f"step {step_str}/{max_steps} ({cur_percent:.2f}%) | "
        f"EVALUATION | "
        f"total time: {format_time(total_time)} | "
        f"eval loss: {eval_loss:.6f}"
    )
    print(eval_str)
    
    # Log to WandB
    wandb.log({
        f'eval/{dataset_abbr}_loss': eval_loss,
        f'eval/{dataset_abbr}_runtime': eval_metrics.get('runtime', 0.0),
        'step': step,
    })


def train_with_accelerate(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: Dict[str, Any],
    deepspeed_config: Optional[Dict[str, Any]] = None,
):
    """
    Unified training loop using Accelerate (works for single-GPU and multi-GPU).
    
    Args:
        model: The model to train
        tokenizer: Tokenizer (for compatibility)
        train_dataset: Training dataset
        eval_dataset: Validation dataset
        config: Training configuration dict with keys:
            - output_dir: Where to save checkpoints
            - max_steps: Total training steps
            - batch_size: Per-device batch size
            - gradient_accumulation_steps: Gradient accumulation steps
            - learning_rate: Learning rate
            - weight_decay: Weight decay
            - adam_beta1, adam_beta2: Adam betas
            - warmup_steps: Warmup steps
            - max_grad_norm: Gradient clipping
            - eval_steps: Evaluation frequency
            - save_steps: Checkpoint save frequency
            - steps_to_save: List of specific steps to save at
            - max_length: Sequence length (for logging)
            - valid_dataset_abbr: Name for validation dataset
            - logging_steps: Logging frequency
            - resume_from_checkpoint: Path to checkpoint to resume from
        deepspeed_config: Optional DeepSpeed configuration dict
    """
    # Extract config
    output_dir = config['output_dir']
    max_steps = config['max_steps']
    batch_size = config['batch_size']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    learning_rate = config['learning_rate']
    weight_decay = config.get('weight_decay', 0.1)
    adam_beta1 = config.get('adam_beta1', 0.95)
    adam_beta2 = config.get('adam_beta2', 0.99)
    warmup_steps = config.get('warmup_steps', 0)
    max_grad_norm = config.get('max_grad_norm', 1.0)
    eval_steps = config.get('eval_steps', 500)
    save_steps = config.get('save_steps', 10000)
    steps_to_save = config.get('steps_to_save', [])
    max_length = config.get('max_length', 4096)
    valid_dataset_abbr = config.get('valid_dataset_abbr', 'valid')
    logging_steps = config.get('logging_steps', 1)
    resume_from_checkpoint = config.get('resume_from_checkpoint', None)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Accelerator (handles DeepSpeed if config is provided)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='bf16',
        deepspeed_plugin=deepspeed_config,
    )
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=weight_decay,
    )

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(max_steps - current_step) / float(max(1, max_steps - warmup_steps)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Streaming dataset handles shuffling
        num_workers=0,
        collate_fn=DefaultDataCollator(return_tensors='pt'),
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=DefaultDataCollator(return_tensors='pt'),
    )
    
    # Prepare everything with Accelerate
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from_checkpoint is not None:
        start_step = load_checkpoint(accelerator, resume_from_checkpoint)
    
    # Training state
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    start_time = datetime.now()
    valid_total_time = 0.0
    batch_token = batch_size * gradient_accumulation_steps * max_length * world_size
    
    if rank == 0:
        print(f"\n***** Running Training with Accelerate *****")
        print(f"  Num processes = {world_size}")
        print(f"  Max steps = {max_steps}")
        print(f"  Batch size per device = {batch_size}")
        print(f"  Gradient accumulation steps = {gradient_accumulation_steps}")
        print(f"  Total optimization batch size = {batch_size * gradient_accumulation_steps * world_size}")
        print(f"  Learning rate = {learning_rate}")
        print(f"  Mixed precision = bf16")
        print(f"  Output directory = {output_dir}\n")
    
    # Training loop
    model.train()
    step = start_step
    train_iter = iter(train_dataloader)
    
    while step < max_steps:
        step_start_time = time.time()
        
        # Gradient accumulation handled by Accelerator context
        total_loss = 0.0
        
        with accelerator.accumulate(model):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            
            # Forward pass (mixed precision handled by Accelerate)
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping and get grad norm BEFORE zero_grad
            grad_norm = 0.0
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss = loss.item()
        
        # Only increment step when we actually do an optimizer step
        if accelerator.sync_gradients:
            step += 1
            
            # Calculate step time
            step_time_ms = (time.time() - step_start_time) * 1000
            
            # Get current learning rate
            current_lr = lr_scheduler.get_last_lr()[0]
            
            # Log training metrics
            log_training_step(
                step=step,
                max_steps=max_steps,
                loss=total_loss,
                grad_norm=grad_norm,
                learning_rate=current_lr,
                base_lr=learning_rate,
                step_time_ms=step_time_ms,
                batch_token=batch_token,
                world_size=world_size,
                start_time=start_time,
                valid_total_time=valid_total_time,
                logging_steps=logging_steps,
            )
            
            # Evaluation
            if step % eval_steps == 0 or step in steps_to_save:
                eval_metrics = evaluate(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    accelerator=accelerator,
                )
                valid_total_time += eval_metrics['runtime']
                
                log_evaluation(
                    step=step,
                    max_steps=max_steps,
                    eval_metrics=eval_metrics,
                    dataset_abbr=valid_dataset_abbr,
                    start_time=start_time,
                )
            
            # Save checkpoint
            if (step % save_steps == 0 or step in steps_to_save) and step > 0:
                save_checkpoint(
                    accelerator=accelerator,
                    train_dataset=train_dataset,
                    step=step,
                    output_dir=output_dir,
                )
    
    if rank == 0:
        end_time = datetime.now()
        total_time = end_time - start_time
        print(f"\n[{str(end_time)}] 100.00% {max_steps} / {max_steps} [{str(total_time)} / {str(total_time)}]")
        print("Training is over!")
    
    return model
