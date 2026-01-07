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
from datasets import load_dataset
from transformers import GPT2Tokenizer
import wandb

sys.path.insert(0, 'src')
from grassmann_flows.models import GrassmannGPT


# -----------------------------------------------------------------------------
# Small Transformer Baseline (size-matched to Grassmann)
# -----------------------------------------------------------------------------

class SmallTransformerBlock(nn.Module):
    """Standard transformer block with multi-head attention."""

    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        normed = self.ln1(x)
        # Generate causal mask if not provided
        if attn_mask is None:
            seq_len = x.size(1)
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device, dtype=x.dtype)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, is_causal=True)
        x = x + self.dropout(attn_out)

        # FFN with residual
        normed = self.ln2(x)
        x = x + self.ffn(normed)

        return x


class SmallTransformer(nn.Module):
    """Size-matched Transformer for fair comparison with Grassmann."""

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 256,
        model_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim

        ff_dim = ff_dim or 4 * model_dim

        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SmallTransformerBlock(model_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=device))
        hidden_states = self.embedding_dropout(tok_emb + pos_emb)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Wikitext-2 Dataset
# -----------------------------------------------------------------------------

class Wikitext2Dataset(Dataset):
    """Wikitext-2 dataset for language modeling."""

    def __init__(self, split: str, tokenizer, max_seq_len: int = 256):
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
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        chunk = self.tokens[start:start + self.max_seq_len]
        x = torch.tensor(chunk, dtype=torch.long)
        return x, x.clone()  # input and target are same for LM


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, log_interval=50):
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, labels=y)
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
    parser = argparse.ArgumentParser(description="Wikitext-2 Paper Reproduction")
    parser.add_argument("--model", type=str, default="both", choices=["grassmann", "transformer", "both"])
    parser.add_argument("--model-dim", type=int, default=256, help="Model dimension (256 gives ~15M params)")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="outputs/wikitext2_reproduction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer)

    # Load datasets
    print("Loading Wikitext-2...")
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
            project="grassmann-flows-wikitext2",
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

        if model_type == "grassmann":
            model = GrassmannGPT(
                vocab_size=vocab_size,
                max_seq_len=args.max_seq_len,
                model_dim=args.model_dim,
                num_layers=args.num_layers,
                reduced_dim=32,  # Paper's value
                ff_dim=4 * args.model_dim,
                window_sizes=[1, 2, 4, 8, 12, 16],  # Paper's values
                dropout=0.1,
            )
        else:
            model = SmallTransformer(
                vocab_size=vocab_size,
                max_seq_len=args.max_seq_len,
                model_dim=args.model_dim,
                num_layers=args.num_layers,
                num_heads=8,
                ff_dim=4 * args.model_dim,
                dropout=0.1,
            )

        model = model.to(device)
        num_params = model.get_num_params()
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

        best_val_loss = float("inf")
        best_val_ppl = float("inf")
        train_losses = []
        val_losses = []

        epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
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
