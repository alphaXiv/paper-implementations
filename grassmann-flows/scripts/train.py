"""
Training script for comparing GPT-2 vs GrassmannGPT.

Usage:
    # Train GPT-2 baseline
    python train.py --model gpt2 --epochs 5

    # Train GrassmannGPT
    python train.py --model grassmann --epochs 5

    # Train both side-by-side (separate runs, shared data)
    python train.py --model both --epochs 5
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from src.grassmann_flows.models import GPT2, GrassmannGPT
from src.grassmann_flows.data import (
    load_wikitext,
    load_openwebtext,
    load_fineweb_edu,
    get_dataloader,
    get_tokenizer,
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_type: str = "gpt2"  # "gpt2" or "grassmann"
    vocab_size: int = 50257
    max_seq_len: int = 256
    model_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8  # For GPT-2
    reduced_dim: int = 64  # For Grassmann
    ff_dim: int = 2048
    dropout: float = 0.1

    # Grassmann-specific
    window_sizes: List[int] = None

    # Data
    dataset: str = "wikitext"  # "wikitext", "openwebtext", "fineweb"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    # Training
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    use_amp: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "outputs"

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [1, 2, 4, 8, 16, 32]


class MetricsLogger:
    """Comprehensive metrics logging."""

    def __init__(self, log_dir: str, model_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard" / model_name))

        # Metrics storage
        self.train_metrics: List[Dict[str, Any]] = []
        self.eval_metrics: List[Dict[str, Any]] = []
        self.epoch_metrics: List[Dict[str, Any]] = []

        # Running stats
        self.step = 0
        self.epoch = 0
        self.total_tokens = 0
        self.start_time = time.time()

    def log_train_step(
        self,
        loss: float,
        lr: float,
        grad_norm: float,
        batch_size: int,
        seq_len: int,
        step_time: float,
    ):
        """Log training step metrics."""
        self.step += 1
        self.total_tokens += batch_size * seq_len

        perplexity = math.exp(min(loss, 20))  # Clamp to avoid overflow
        tokens_per_sec = batch_size * seq_len / step_time

        metrics = {
            "step": self.step,
            "epoch": self.epoch,
            "loss": loss,
            "perplexity": perplexity,
            "learning_rate": lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": tokens_per_sec,
            "total_tokens": self.total_tokens,
            "step_time": step_time,
            "wall_time": time.time() - self.start_time,
        }

        self.train_metrics.append(metrics)

        # TensorBoard
        self.writer.add_scalar("train/loss", loss, self.step)
        self.writer.add_scalar("train/perplexity", perplexity, self.step)
        self.writer.add_scalar("train/learning_rate", lr, self.step)
        self.writer.add_scalar("train/grad_norm", grad_norm, self.step)
        self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.step)

        return metrics

    def log_eval(
        self,
        loss: float,
        num_samples: int,
    ):
        """Log evaluation metrics."""
        perplexity = math.exp(min(loss, 20))

        metrics = {
            "step": self.step,
            "epoch": self.epoch,
            "val_loss": loss,
            "val_perplexity": perplexity,
            "num_samples": num_samples,
            "wall_time": time.time() - self.start_time,
        }

        self.eval_metrics.append(metrics)

        # TensorBoard
        self.writer.add_scalar("eval/loss", loss, self.step)
        self.writer.add_scalar("eval/perplexity", perplexity, self.step)

        return metrics

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        epoch_time: float,
    ):
        """Log epoch-level metrics."""
        self.epoch = epoch

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_perplexity": math.exp(min(train_loss, 20)),
            "val_loss": val_loss,
            "val_perplexity": math.exp(min(val_loss, 20)),
            "epoch_time": epoch_time,
            "total_tokens": self.total_tokens,
            "wall_time": time.time() - self.start_time,
        }

        self.epoch_metrics.append(metrics)

        return metrics

    def save(self):
        """Save all metrics to JSON files."""
        with open(self.log_dir / f"{self.model_name}_train_metrics.json", "w") as f:
            json.dump(self.train_metrics, f, indent=2)

        with open(self.log_dir / f"{self.model_name}_eval_metrics.json", "w") as f:
            json.dump(self.eval_metrics, f, indent=2)

        with open(self.log_dir / f"{self.model_name}_epoch_metrics.json", "w") as f:
            json.dump(self.epoch_metrics, f, indent=2)

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


def create_model(config: TrainingConfig) -> nn.Module:
    """Create model based on config."""
    if config.model_type == "gpt2":
        model = GPT2(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        )
    elif config.model_type == "grassmann":
        model = GrassmannGPT(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            reduced_dim=config.reduced_dim,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            window_sizes=config.window_sizes,
        )
        # Exact paper architecture
        model = GrassmannGPT(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            reduced_dim=32,  # Paper uses r=32
            ff_dim=config.ff_dim,
            window_sizes=[1, 2, 4, 8, 12, 16],  # Paper's window sizes
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    return model


def get_lr_scheduler(optimizer, config: TrainingConfig, total_steps: int):
    """Create learning rate scheduler with warmup and cosine decay."""
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        # Cosine decay
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches: Optional[int] = None):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_samples = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, labels=labels)
        total_loss += loss.item() * input_ids.size(0)
        total_samples += input_ids.size(0)

    model.train()
    return total_loss / total_samples, total_samples


def train_model(config: TrainingConfig):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"Training {config.model_type.upper()}")
    print(f"{'='*60}\n")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.model_type}_{timestamp}"
    output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Setup device
    device = torch.device(config.device)
    print(f"Device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer("gpt2")
    config.vocab_size = len(tokenizer)

    # Load data
    print("Loading dataset...")
    if config.dataset == "wikitext":
        train_dataset = load_wikitext(
            tokenizer,
            seq_len=config.max_seq_len,
            version="wikitext-103-raw-v1",
            split="train",
        )
        val_dataset = load_wikitext(
            tokenizer,
            seq_len=config.max_seq_len,
            version="wikitext-103-raw-v1",
            split="validation",
        )
    elif config.dataset == "openwebtext":
        train_dataset = load_openwebtext(
            tokenizer,
            seq_len=config.max_seq_len,
            max_samples=config.max_train_samples,
            split="train",
        )
        # Use subset for validation
        val_dataset = load_wikitext(
            tokenizer,
            seq_len=config.max_seq_len,
            version="wikitext-2-raw-v1",
            split="validation",
        )
    elif config.dataset == "fineweb":
        train_dataset = load_fineweb_edu(
            tokenizer,
            seq_len=config.max_seq_len,
            max_samples=config.max_train_samples,
        )
        val_dataset = load_wikitext(
            tokenizer,
            seq_len=config.max_seq_len,
            version="wikitext-2-raw-v1",
            split="validation",
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Val sequences: {len(val_dataset):,}")

    train_loader = get_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)

    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler
    total_steps = len(train_loader) * config.epochs
    scheduler = get_lr_scheduler(optimizer, config, total_steps)

    # AMP scaler
    scaler = GradScaler() if config.use_amp and device.type == "cuda" else None

    # Metrics logger
    logger = MetricsLogger(str(output_dir), config.model_type)

    # Log model info
    logger.writer.add_text("config", json.dumps(asdict(config), indent=2))
    logger.writer.add_scalar("model/num_params", num_params, 0)

    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Total steps: {total_steps:,}")
    print(f"Logging to: {output_dir}\n")

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(config.epochs):
        epoch_start = time.time()
        epoch_loss = 0
        epoch_steps = 0

        model.train()

        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            if scaler is not None:
                with autocast():
                    _, loss = model(input_ids, labels=labels)
            else:
                _, loss = model(input_ids, labels=labels)

            # Backward pass
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()

            scheduler.step()

            step_time = time.time() - step_start
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Log training metrics
            if global_step % config.log_interval == 0:
                metrics = logger.log_train_step(
                    loss=loss.item(),
                    lr=scheduler.get_last_lr()[0],
                    grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    batch_size=input_ids.size(0),
                    seq_len=input_ids.size(1),
                    step_time=step_time,
                )
                print(
                    f"[Epoch {epoch+1}/{config.epochs}] "
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"PPL: {metrics['perplexity']:.2f} | "
                    f"LR: {metrics['learning_rate']:.2e} | "
                    f"Tok/s: {metrics['tokens_per_sec']:.0f}"
                )

            # Evaluate
            if global_step % config.eval_interval == 0:
                val_loss, val_samples = evaluate(model, val_loader, device, max_batches=50)
                metrics = logger.log_eval(val_loss, val_samples)
                print(
                    f"  >> Validation | Loss: {metrics['val_loss']:.4f} | "
                    f"PPL: {metrics['val_perplexity']:.2f}"
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": asdict(config),
                        "step": global_step,
                        "val_loss": val_loss,
                    }, output_dir / "best_model.pt")
                    print(f"  >> Saved new best model (val_loss: {val_loss:.4f})")

            # Save checkpoint
            if global_step % config.save_interval == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": asdict(config),
                    "step": global_step,
                    "epoch": epoch,
                }, output_dir / f"checkpoint_step{global_step}.pt")

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / epoch_steps

        # Full validation
        val_loss, val_samples = evaluate(model, val_loader, device)
        epoch_metrics = logger.log_epoch(epoch + 1, avg_train_loss, val_loss, epoch_time)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs} Complete")
        print(f"  Train Loss: {epoch_metrics['train_loss']:.4f} | Train PPL: {epoch_metrics['train_perplexity']:.2f}")
        print(f"  Val Loss: {epoch_metrics['val_loss']:.4f} | Val PPL: {epoch_metrics['val_perplexity']:.2f}")
        print(f"  Epoch Time: {epoch_time:.1f}s | Total Tokens: {logger.total_tokens:,}")
        print(f"{'='*60}\n")

        logger.save()

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "final_val_loss": val_loss,
    }, output_dir / "final_model.pt")

    logger.save()
    logger.close()

    print(f"\nTraining complete! Results saved to: {output_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {math.exp(best_val_loss):.2f}")

    return output_dir, best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 vs GrassmannGPT")

    # Model selection
    parser.add_argument(
        "--model", type=str, default="both",
        help="Which model(s) to train"
    )

    # Model architecture
    parser.add_argument("--model-dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads (GPT-2)")
    parser.add_argument("--reduced-dim", type=int, default=64, help="Reduced dimension (Grassmann)")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Maximum sequence length")

    # Data
    parser.add_argument(
        "--dataset", type=str, default="wikitext",
        choices=["wikitext", "openwebtext", "fineweb"],
        help="Dataset to use"
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples")

    # Training
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")

    # Logging
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate every N steps")

    args = parser.parse_args()

    # Create base config
    base_config = TrainingConfig(
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        reduced_dim=args.reduced_dim,
        max_seq_len=args.max_seq_len,
        dataset=args.dataset,
        max_train_samples=args.max_train_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
    )

    results = {}

    if args.model in ["gpt2", "both"]:
        config = TrainingConfig(**{**asdict(base_config), "model_type": "gpt2"})
        output_dir, val_loss = train_model(config)
        results["gpt2"] = {"output_dir": str(output_dir), "val_loss": val_loss}

    if args.model in ["grassmann", "both"]:
        config = TrainingConfig(**{**asdict(base_config), "model_type": "grassmann"})
        output_dir, val_loss = train_model(config)
        results["grassmann"] = {"output_dir": str(output_dir), "val_loss": val_loss}

        output_dir, val_loss = train_model(config)

    # Print comparison summary
    if args.model == "both":
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"GPT-2 best val loss: {results['gpt2']['val_loss']:.4f} (PPL: {math.exp(results['gpt2']['val_loss']):.2f})")
        print(f"Grassmann best val loss: {results['grassmann']['val_loss']:.4f} (PPL: {math.exp(results['grassmann']['val_loss']):.2f})")
        print("="*60)

    # Save comparison results
    with open(Path(args.output_dir) / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
