"""
SNLI Classification Training Script

Paper specification (Section 4.2):
- DistilBERT-base-uncased backbone
- 2 classification heads: Transformer vs Grassmann-Plucker
- 20 epochs training
- Train/val splits during training, test split for final evaluation
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import DistilBertTokenizer
from datasets import load_dataset
import wandb
from tqdm import tqdm

sys.path.insert(0, 'src')
from attn_is_not_all_you_need.models import SNLIModel
from attn_is_not_all_you_need.data import SNLIDataset


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler=None, log_interval=50):
    """Train SNLI model for one epoch.
    
    Args:
        model: The SNLI model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run on
        epoch: Current epoch number
        scaler: GradScaler for AMP (optional)
        log_interval: Logging interval
        
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0
    use_amp = scaler is not None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        # Use automatic mixed precision
        with autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += input_ids.size(0)
        
        if step % log_interval == 0:
            acc = total_correct / total_count if total_count > 0 else 0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}",
                "grad_norm": f"{grad_norm.item():.4f}",
            })
    
    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate SNLI classification model.
    
    Args:
        model: The SNLI model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run on
        
    Returns:
        tuple: (accuracy, avg_loss)
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        logits = outputs["logits"]
        
        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += input_ids.size(0)
    
    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count
    return accuracy, avg_loss


def main():
    """Main training function for SNLI."""
    parser = argparse.ArgumentParser(description="SNLI Classification Training")
    parser.add_argument("--model", type=str, default="both", choices=["grassmann", "transformer", "both"])
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (paper: 20)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="outputs/snli_reproduction")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load datasets - use train/val during training, test is for final eval
    print("Loading SNLI...")
    train_dataset = SNLIDataset("train", tokenizer, max_seq_len=48)
    val_dataset = SNLIDataset("validation", tokenizer, max_seq_len=48)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    results = {}
    
    models_to_train = []
    if args.model in ["grassmann", "both"]:
        models_to_train.append("grassmann")
    if args.model in ["transformer", "both"]:
        models_to_train.append("transformer")
    
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training SNLI: {model_type.upper()}")
        print(f"{'='*60}")
        
        # Initialize W&B
        wandb.init(
            project="attn-is-not-all-you-need-snli",
            name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_type": model_type,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
            },
            dir=str(output_dir),
        )
        
        # Paper specs for SNLI heads
        head_kwargs = {
            "num_heads": 4,
            "ff_dim": 512,
            "dropout": 0.1,
            "num_classes": 3,
        }
        
        if model_type == "grassmann":
            head_kwargs.update({
                "dmodel": 256,
                "dproj": 64,
                "window_size": 8,
                "stride": 8,
                "num_layers": 2,
            })
            model = SNLIModel(
                head_type="grassmann",
                freeze_backbone=True,
                **head_kwargs
            )
        else:
            head_kwargs.update({
                "model_dim": 256,
                "num_layers": 2,
            })
            model = SNLIModel(
                head_type="transformer",
                freeze_backbone=True,
                **head_kwargs
            )
        
        model = model.to(device)
        
        # Compile model with torch.compile for better performance
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        
        num_params = model.get_num_params()
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        # Enable automatic mixed precision for Flash Attention
        use_amp = torch.cuda.is_available()
        scaler = GradScaler('cuda', enabled=use_amp)
        if use_amp:
            print("Automatic Mixed Precision (AMP) enabled - Flash Attention will be used!")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        
        # Create checkpoint directory
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        
        epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, scaler if use_amp else None)
            val_acc, val_loss = evaluate(model, val_loader, device)
            
            epoch_pbar.set_postfix({
                "epoch": epoch,
                "train_acc": f"{train_acc:.4f}",
                "val_acc": f"{val_acc:.4f}",
            })
            
            print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log to W&B
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_loss": val_loss,
            })
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")
        
        print(f"\nFinal Results for {model_type.upper()}:")
        print(f"  Training complete. Checkpoints saved in {ckpt_dir}")
        print(f"  Note: Run eval_snli.py with checkpoint for test accuracy")
        
        results[model_type] = {
            "num_params": num_params,
            "final_val_acc": val_acc,
            "final_val_loss": val_loss,
        }
        
        # Log final results to W&B
        wandb.log({
            "final/val_acc": val_acc,
            "final/val_loss": val_loss,
            "final/num_params": num_params,
        })
        
        wandb.finish()
    
    # Save results
    with open(output_dir / "snli_results.json", "w") as f:
        save_results = {}
        for k, v in results.items():
            save_results[k] = {
                "num_params": v["num_params"],
                "final_val_acc": float(v["final_val_acc"]),
                "final_val_loss": float(v["final_val_loss"]),
            }
        json.dump(save_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training complete! Checkpoints saved.")
    print("Run separate test evaluation with:")
    print(f"  python -m attn_is_not_all_you_need.eval_snli --checkpoint {output_dir}/checkpoints/epoch_<N>.pt --model_type <model> --split test")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    main()
