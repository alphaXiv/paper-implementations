"""
Exact Paper Reproduction: Wikitext-2 Training

Reproduces arXiv 2512.19428 exactly:
- Dataset: Wikitext-2
- Model size: 13-18M parameters
- Compares Grassmann vs size-matched Transformer
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import BertTokenizer, DistilBertModel, DistilBertTokenizer
import wandb
from tqdm import tqdm

sys.path.insert(0, 'src')
from attn_is_not_all_you_need.models import GrassmannGPT, SNLIModel
from attn_is_not_all_you_need.models.gpt2 import BaseTransformer



# -----------------------------------------------------------------------------
# Wikitext-2 Dataset
# -----------------------------------------------------------------------------

class Wikitext2Dataset(Dataset):
    """Wikitext-2 dataset for language modeling."""

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
        all_text = "\n".join([t for t in dataset["text"] if t.strip()])
        self.tokens = tokenizer.encode(all_text)

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
        """
        start = idx * self.max_seq_len
        chunk = self.tokens[start:start + self.max_seq_len]
        x = torch.tensor(chunk, dtype=torch.long)
        return x, x.clone()  # input and target are same for LM


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler=None, log_interval=50):
    """Train for one epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader for training data.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to run on.
        epoch (int): Current epoch number.
        scaler: GradScaler for AMP (optional).
        log_interval (int): Logging interval.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        
        # Use automatic mixed precision
        with autocast(enabled=use_amp, dtype=torch.bfloat16):
            _, loss = model(x, labels=y)
        
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

        total_loss += loss.item() * x.size(0)
        total_tokens += x.numel()

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{loss.exp().item():.2f}",
                "tok/s": f"{tok_per_sec:.0f}",
                "grad_norm": f"{grad_norm.item():.4f}",
            })

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate the model on validation/test set.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for evaluation data.
        device: Device to run on.

    Returns:
        tuple: (avg_loss, perplexity)
    """
    model.eval()
    total_loss = 0
    total_count = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, labels=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    avg_loss = total_loss / total_count
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()


def main():
    """Main training function for reproducing the paper's experiments."""
    parser = argparse.ArgumentParser(description="Paper Reproduction: Wikitext-2 & SNLI")
    parser.add_argument("--task", type=str, default="wikitext", choices=["wikitext", "snli"], help="Task to train on")
    parser.add_argument("--model", type=str, default="both", choices=["grassmann", "transformer", "both"])
    parser.add_argument("--model-dim", type=int, default=256, help="Model dimension (paper: 256)")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers (paper: 6 or 12)")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Max sequence length (paper: 128 or 256)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto: 32 for L=128, 16 for L=256)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (auto: 30 for wikitext, 20 for SNLI)")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    # Set defaults based on task and block size
    if args.batch_size is None:
        args.batch_size = 32 if args.max_seq_len == 128 else 16
    if args.epochs is None:
        args.epochs = 30 if args.task == "wikitext" else 20
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.task}_reproduction"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "wikitext":
        return train_wikitext(args, device, output_dir)
    else:
        # Call separate SNLI training script
        import subprocess
        cmd = [
            "python", "-m", "attn_is_not_all_you_need.train_snli",
            "--model", args.model,
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--output-dir", str(output_dir),
        ]
        result = subprocess.run(cmd, cwd=".")
        return result.returncode


def train_wikitext(args, device, output_dir):
    """Train on Wikitext-2 language modeling."""
    # Load tokenizer - use BERT tokenizer for vocab ~30522
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = len(tokenizer)  # Should be ~30522

    # Load datasets
    print(f"Loading Wikitext-2 (block_size={args.max_seq_len})...")
    train_dataset = Wikitext2Dataset("train", tokenizer, args.max_seq_len)
    val_dataset = Wikitext2Dataset("validation", tokenizer, args.max_seq_len)
    test_dataset = Wikitext2Dataset("test", tokenizer, args.max_seq_len)

    print(f"Train: {len(train_dataset)} chunks, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = {}

    models_to_train = []
    if args.model in ["grassmann", "both"]:
        models_to_train.append("grassmann")
    if args.model in ["transformer", "both"]:
        models_to_train.append("transformer")

    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training: {model_type.upper()}")
        print(f"{'='*60}")

        # Initialize W&B for this model
        wandb.init(
            project="attn-is-not-all-you-need-wikitext2",
            name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_type": model_type,
                "model_dim": args.model_dim,
                "num_layers": args.num_layers,
                "max_seq_len": args.max_seq_len,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
            },
            dir=str(output_dir),
        )

        # Paper: W={1,2,4,8,12,16} for 6-layer, (1,1,2,2,4,4,8,8,12,12,16,16) for 12-layer
        if args.num_layers == 6:
            window_sizes = [1, 2, 4, 8, 12, 16]
        elif args.num_layers == 12:
            window_sizes = [1, 1, 2, 2, 4, 4, 8, 8, 12, 12, 16, 16]
        else:
            window_sizes = [1, 2, 4, 8, 12, 16]  # default
        
        if model_type == "grassmann":
            model = GrassmannGPT(
                vocab_size=vocab_size,
                max_seq_len=args.max_seq_len,
                model_dim=args.model_dim,
                num_layers=args.num_layers,
                reduced_dim=32,  # Paper's value
                ff_dim=1024,  # Paper: dff=1024
                window_sizes=window_sizes,
                dropout=0.1,
            )
        else:
            model = BaseTransformer(
                vocab_size=vocab_size,
                max_seq_len=args.max_seq_len,
                model_dim=args.model_dim,
                num_layers=args.num_layers,
                num_heads=4,  # Paper: 4 heads
                ff_dim=1024,  # Paper: dff=1024
                dropout=0.1,
            )

        model = model.to(device)
        
        # Compile model with torch.compile for better performance
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

        # Enable automatic mixed precision for Flash Attention
        use_amp = torch.cuda.is_available()
        scaler = GradScaler(enabled=use_amp)
        if use_amp:
            print("Automatic Mixed Precision (AMP) enabled - Flash Attention will be used!")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

        best_val_loss = float("inf")
        best_val_ppl = float("inf")
        train_losses = []
        val_losses = []

        epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, scaler if use_amp else None)
            val_loss, val_ppl = evaluate(model, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            epoch_pbar.set_postfix({
                "epoch": epoch,
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_ppl": f"{val_ppl:.2f}",
            })

            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

            # Log to W&B
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_ppl
                torch.save(model.state_dict(), output_dir / f"{model_type}_best.pt")

        # Final test evaluation
        model.load_state_dict(torch.load(output_dir / f"{model_type}_best.pt"))
        test_loss, test_ppl = evaluate(model, test_loader, device)

        print(f"\nFinal Results for {model_type.upper()}:")
        print(f"  Best Val Loss: {best_val_loss:.4f}, Best Val PPL: {best_val_ppl:.2f}")
        print(f"  Test Loss: {test_loss:.4f}, Test PPL: {test_ppl:.2f}")

        results[model_type] = {
            "num_params": num_params,
            "best_val_loss": best_val_loss,
            "best_val_ppl": best_val_ppl,
            "test_loss": test_loss,
            "test_ppl": test_ppl,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        # Log final results to W&B
        wandb.log({
            "final/best_val_loss": best_val_loss,
            "final/best_val_perplexity": best_val_ppl,
            "final/test_loss": test_loss,
            "final/test_perplexity": test_ppl,
            "final/num_params": num_params,
        })

        # Finish W&B run for this model
        wandb.finish()

    # Save results
    with open(output_dir / "results.json", "w") as f:
        # Convert non-serializable items
        save_results = {}
        for k, v in results.items():
            save_results[k] = {
                "num_params": v["num_params"],
                "best_val_loss": float(v["best_val_loss"]),
                "best_val_ppl": float(v["best_val_ppl"]),
                "test_loss": float(v["test_loss"]),
                "test_ppl": float(v["test_ppl"]),
            }
        json.dump(save_results, f, indent=2)

    # Print comparison
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON: Paper Reproduction on Wikitext-2")
        print(f"{'='*60}")

        g = results["grassmann"]
        t = results["transformer"]

        print(f"{'Model':<20} {'Params':<12} {'Val PPL':<12} {'Test PPL':<12}")
        print("-" * 56)
        print(f"{'Grassmann':<20} {g['num_params']/1e6:.2f}M{'':<6} {g['best_val_ppl']:<12.2f} {g['test_ppl']:<12.2f}")
        print(f"{'Transformer':<20} {t['num_params']/1e6:.2f}M{'':<6} {t['best_val_ppl']:<12.2f} {t['test_ppl']:<12.2f}")
        print("-" * 56)

        ppl_ratio = g["test_ppl"] / t["test_ppl"]
        gap_percent = (ppl_ratio - 1) * 100
        print(f"\nGrassmann/Transformer PPL ratio: {ppl_ratio:.3f}")
        print(f"Gap: {gap_percent:.1f}% (Paper claims 10-15%)")

        if gap_percent <= 15:
            print("RESULT: Paper claim VERIFIED - within 15% gap")
        else:
            print(f"RESULT: Paper claim NOT verified - gap is {gap_percent:.1f}%")


if __name__ == "__main__":
    main()
