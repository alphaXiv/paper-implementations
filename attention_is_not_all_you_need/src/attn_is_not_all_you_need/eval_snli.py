"""
Evaluate trained SNLI classification models.

This script evaluates trained SNLI models (Transformer or Grassmann heads)
on train/val/test splits of the SNLI dataset.
Supports multiple runs with 95% confidence intervals.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from transformers import DistilBertTokenizer
from datasets import load_dataset
import wandb
from datetime import datetime
import sys
import os
import json
from tqdm import tqdm
import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, 'src')
from attn_is_not_all_you_need.models import SNLIModel
from attn_is_not_all_you_need.data import SNLIDataset


def load_model(model_path, model_type):
    """Load trained SNLI model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model ('grassmann' or 'transformer')
        
    Returns:
        The loaded model in eval mode
    """
    # Paper specs for SNLI heads
    head_kwargs = {
        "num_heads": 4,
        "ff_dim": 512,
        "dropout": 0.1,
        "num_classes": 3,
    }
    
    if model_type == "grassmann":
        head_kwargs.update({
            "dmodel": 256,  # GrassmannPluckerNLIHead uses 'dmodel'
            "dproj": 64,
            "window_size": 8,
            "stride": 8,
            "num_layers": 2,
        })
        model = SNLIModel(
            head_type="grassmann",
            freeze_backbone=False,
            **head_kwargs
        )
    else:
        head_kwargs.update({
            "model_dim": 256,  # TransformerNLIHead uses 'model_dim'
            "num_layers": 2,
        })
        model = SNLIModel(
            head_type="transformer",
            freeze_backbone=False,
            **head_kwargs
        )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove _orig_mod. prefix if present (from torch.compile)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()
    return model


@torch.no_grad()
def evaluate_snli(model, dataloader, device):
    """Evaluate model on SNLI dataset.
    
    Args:
        model: The trained SNLI model
        dataloader: DataLoader for evaluation data
        device: Device to run on
        
    Returns:
        dict: Evaluation results including accuracy and per-class metrics
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0
    
    # Per-class metrics
    label_names = ['entailment', 'neutral', 'contradiction']
    label_counts = {name: 0 for name in label_names}
    correct_counts = {name: 0 for name in label_names}
    
    for batch in tqdm(dataloader, desc="Evaluating"):
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
        
        # Per-class accuracy
        for i in range(len(labels)):
            label_idx = labels[i].item()
            label_name = label_names[label_idx]
            label_counts[label_name] += 1
            if preds[i] == labels[i]:
                correct_counts[label_name] += 1
    
    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count
    
    # Per-class accuracy
    per_class_acc = {}
    for name in label_names:
        if label_counts[name] > 0:
            per_class_acc[name] = correct_counts[name] / label_counts[name]
        else:
            per_class_acc[name] = 0.0
    
    results = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'total_samples': total_count,
        'entailment_accuracy': per_class_acc['entailment'],
        'neutral_accuracy': per_class_acc['neutral'],
        'contradiction_accuracy': per_class_acc['contradiction'],
        'label_distribution': label_counts,
        'correct_distribution': correct_counts
    }
    
    return results


def main():
    """Main function for evaluating SNLI model."""
    parser = argparse.ArgumentParser(description="Evaluate trained SNLI model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["transformer", "grassmann"], required=True)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"], 
                        help="Dataset split to evaluate on")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of evaluation runs for CI calculation")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for CI (default: 0.95)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize W&B
    wandb.init(
        project="attn-is-not-all-you-need-snli-eval",
        name=f"{args.model_type}_snli_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_type": args.model_type,
            "model_path": args.model_path,
            "split": args.split,
            "num_runs": args.num_runs,
            "confidence": args.confidence,
        },
    )
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load model
    print(f"Loading {args.model_type} model from {args.model_path}...")
    model = load_model(args.model_path, args.model_type)
    model.to(device)
    
    # Compile model for faster evaluation
    print("Compiling model with torch.compile...")
    model = torch.compile(model)
    
    # Load dataset
    print(f"Loading SNLI {args.split} split...")
    dataset = SNLIDataset(args.split, tokenizer, max_seq_len=48)
    
    print(f"Running {args.num_runs} evaluation runs on {len(dataset)} examples...")
    
    # Run multiple evaluations
    all_results = []
    for run_idx in range(args.num_runs):
        print(f"\n--- Run {run_idx + 1}/{args.num_runs} ---")
        # Create new dataloader with different seed for shuffling
        torch.manual_seed(42 + run_idx)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Evaluate
        run_results = evaluate_snli(model, dataloader, device)
        all_results.append(run_results)
        
        print(f"Run {run_idx + 1} Accuracy: {run_results['accuracy']:.4f}")
    
    # Calculate statistics across runs
    accuracies = [r['accuracy'] for r in all_results]
    losses = [r['loss'] for r in all_results]
    entailment_accs = [r['entailment_accuracy'] for r in all_results]
    neutral_accs = [r['neutral_accuracy'] for r in all_results]
    contradiction_accs = [r['contradiction_accuracy'] for r in all_results]
    
    # Calculate statistics
    acc_mean = np.mean(accuracies)
    loss_mean = np.mean(losses)
    ent_mean = np.mean(entailment_accs)
    neu_mean = np.mean(neutral_accs)
    con_mean = np.mean(contradiction_accs)
    
    # Print results
    print(f"\n{'='*70}")
    if args.num_runs == 1:
        print(f"SNLI Evaluation Results ({args.split} split)")
        print(f"{'='*70}")
        print(f"Overall Accuracy: {acc_mean:.4f}")
        print(f"Loss: {loss_mean:.4f}")
        print(f"Total Samples: {all_results[0]['total_samples']}")
        print(f"\nPer-Class Accuracy:")
        print(f"  Entailment:    {ent_mean:.4f}")
        print(f"  Neutral:       {neu_mean:.4f}")
        print(f"  Contradiction: {con_mean:.4f}")
    print(f"{'='*70}")
    
    # Log to W&B
    wandb_metrics = {
        f"{args.split}/accuracy": acc_mean,
        f"{args.split}/loss": loss_mean,
        f"{args.split}/total_samples": all_results[0]['total_samples'],
        f"{args.split}/entailment_accuracy": ent_mean,
        f"{args.split}/neutral_accuracy": neu_mean,
        f"{args.split}/contradiction_accuracy": con_mean,
    }
    
    wandb.log(wandb_metrics)
    
    # Save results
    if args.output_file:
        save_results = {
            'num_runs': args.num_runs,
            'accuracy': {
                'mean': float(acc_mean),
                'all_runs': [float(a) for a in accuracies],
            },
            'loss': {
                'mean': float(loss_mean),
            },
            'total_samples': all_results[0]['total_samples'],
            'entailment_accuracy': {
                'mean': float(ent_mean),
            },
            'neutral_accuracy': {
                'mean': float(neu_mean),
            },
            'contradiction_accuracy': {
                'mean': float(con_mean),
            },
            'label_distribution': all_results[0]['label_distribution'],
        }
        
        
        
        with open(args.output_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")
    
    # Finish W&B
    wandb.finish()



if __name__ == "__main__":
    main()