"""
Evaluate trained Wikitext-2 language models on test split.

This script evaluates trained language models on the Wikitext-2 test set
and reports perplexity with 95% confidence intervals.
"""

import torch
import argparse
from datetime import datetime
from transformers import BertTokenizer
from datasets import load_dataset
import json
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
from scipy import stats
import wandb

sys.path.insert(0, 'src')
from attn_is_not_all_you_need.models import GrassmannGPT
from attn_is_not_all_you_need.models.gpt2 import BaseTransformer
from attn_is_not_all_you_need.data import Wikitext2Dataset


def load_model(model_path, model_type, vocab_size, max_seq_len, model_dim, num_layers):
    """Load trained model from checkpoint."""
    # Paper: W={1,2,4,8,12,16} for 6-layer, repeated for 12-layer
    if num_layers == 6:
        window_sizes = [1, 2, 4, 8, 12, 16]
    elif num_layers == 12:
        window_sizes = [1, 1, 2, 2, 4, 4, 8, 8, 12, 12, 16, 16]
    else:
        window_sizes = [1, 2, 4, 8, 12, 16]
    
    if model_type == "grassmann":
        model = GrassmannGPT(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            reduced_dim=32,
            ff_dim=1024,
            window_sizes=window_sizes,
            dropout=0.1,
        )
    else:
        model = BaseTransformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=4,
            ff_dim=1024,
            dropout=0.1,
        )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate the model on test set.
    
    Returns:
        tuple: (avg_loss, perplexity)
    """
    model.eval()
    total_loss = 0
    total_count = 0

    for x, y in tqdm(dataloader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        _, loss = model(x, labels=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    avg_loss = total_loss / total_count
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wikitext-2 models on test split")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, required=True, choices=["transformer", "grassmann"])
    parser.add_argument("--max-seq-len", type=int, default=128, help="Sequence length (128 or 256)")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers (6 or 12)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of evaluation runs for CI calculation")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for CI (default: 0.95)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize W&B
    wandb.init(
        project="attn-is-not-all-you-need-wikitext2-eval",
        name=f"{args.model_type}_wikitext2_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_type": args.model_type,
            "model_path": args.model_path,
            "max_seq_len": args.max_seq_len,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "num_runs": args.num_runs,
            "confidence": args.confidence,
        },
    )

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = len(tokenizer)

    # Load model
    print(f"Loading {args.model_type} model from {args.model_path}...")
    model = load_model(
        args.model_path,
        args.model_type,
        vocab_size,
        args.max_seq_len,
        256,  # model_dim
        args.num_layers
    )
    model.to(device)
    
    # Compile model for faster evaluation
    print("Compiling model with torch.compile...")
    model = torch.compile(model)

    # Load test dataset
    print(f"Loading Wikitext-2 test split (L={args.max_seq_len})...")
    test_dataset = Wikitext2Dataset("test", tokenizer, args.max_seq_len)

    print(f"Running {args.num_runs} evaluation runs on {len(test_dataset)} examples...")

    # Run multiple evaluations
    all_losses = []
    all_ppls = []
    for run_idx in range(args.num_runs):
        print(f"\n--- Run {run_idx + 1}/{args.num_runs} ---")
        # Create dataloader (deterministic order)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Evaluate
        test_loss, test_ppl = evaluate(model, test_loader, device)
        all_losses.append(test_loss)
        all_ppls.append(test_ppl)
        
        print(f"Run {run_idx + 1} - Loss: {test_loss:.4f}, Perplexity: {test_ppl:.2f}")

    # Calculate statistics
    def calculate_ci(values, confidence=0.95):
        """Calculate mean and confidence interval."""
        n = len(values)
        mean = np.mean(values)
        std_err = stats.sem(values)
        ci = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean, ci, std_err

    loss_mean, loss_ci, loss_std_err = calculate_ci(all_losses, args.confidence)
    ppl_mean, ppl_ci, ppl_std_err = calculate_ci(all_ppls, args.confidence)

    # Print results with CI
    print(f"\n{'='*70}")
    print(f"Wikitext-2 Test Results - {args.num_runs} runs")
    print(f"{'='*70}")
    print(f"Loss:       {loss_mean:.4f} ± {loss_ci:.4f} (95% CI: [{loss_mean - loss_ci:.4f}, {loss_mean + loss_ci:.4f}])")
    print(f"Perplexity: {ppl_mean:.2f} ± {ppl_ci:.2f} (95% CI: [{ppl_mean - ppl_ci:.2f}, {ppl_mean + ppl_ci:.2f}])")
    print(f"{'='*70}")

    # Save results
    results = {
        "model_type": args.model_type,
        "max_seq_len": args.max_seq_len,
        "num_layers": args.num_layers,
        "num_runs": args.num_runs,
        "confidence_level": args.confidence,
        "test_loss": {
            "mean": float(loss_mean),
            "ci": float(loss_ci),
            "std_err": float(loss_std_err),
            "ci_lower": float(loss_mean - loss_ci),
            "ci_upper": float(loss_mean + loss_ci),
            "all_runs": [float(l) for l in all_losses],
        },
        "test_perplexity": {
            "mean": float(ppl_mean),
            "ci": float(ppl_ci),
            "std_err": float(ppl_std_err),
            "ci_lower": float(ppl_mean - ppl_ci),
            "ci_upper": float(ppl_mean + ppl_ci),
            "all_runs": [float(p) for p in all_ppls],
        },
    }

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    # Log to W&B
    wandb.log({
        "test/loss_mean": loss_mean,
        "test/loss_ci": loss_ci,
        "test/perplexity_mean": ppl_mean,
        "test/perplexity_ci": ppl_ci,
    })
    
    wandb.finish()



if __name__ == "__main__":
    main()
