"""
Evaluate trained language models on SNLI (Stanford Natural Language Inference).

This script loads a trained model and evaluates its zero-shot performance on SNLI
using prompting-based natural language inference.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from transformers import GPT2Tokenizer
from datasets import load_dataset
import wandb
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, 'src')
from grassmann_flows.models import GrassmannGPT, GPT2


def load_model(model_path, model_type, vocab_size, max_seq_len, model_dim, num_layers):
    """Load trained model from checkpoint."""
    if model_type == "grassmann":
        model = GrassmannGPT(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            reduced_dim=32,
            ff_dim=4 * model_dim,
            window_sizes=[1, 2, 4, 8, 12, 16],
            dropout=0.1,
        )
    else:  # gpt2
        model = GPT2(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=12,
            ff_dim=4 * model_dim,
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


def create_prompt(premise, hypothesis):
    """Create a prompt for NLI classification."""
    return f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? Answer with 'yes', 'no', or 'maybe':\n"


def get_prediction(logits, tokenizer):
    """Get prediction from model logits."""
    # Get the last token logits
    next_token_logits = logits[:, -1, :]

    # Get probabilities for yes/no/maybe tokens
    vocab = tokenizer.get_vocab()
    yes_token = vocab.get('yes', vocab.get(' Yes', None))
    no_token = vocab.get('no', vocab.get(' No', None))
    maybe_token = vocab.get('maybe', vocab.get(' Maybe', vocab.get('neutral', None)))

    if yes_token is None or no_token is None:
        # Fallback: use argmax
        pred_token = torch.argmax(next_token_logits, dim=-1).item()
        pred_text = tokenizer.decode(pred_token).lower().strip()
        if 'yes' in pred_text or 'entail' in pred_text:
            return 'entailment'
        elif 'no' in pred_text or 'contradict' in pred_text:
            return 'contradiction'
        else:
            return 'neutral'
    else:
        probs = torch.softmax(next_token_logits, dim=-1)
        yes_prob = probs[0, yes_token].item() if yes_token else 0
        no_prob = probs[0, no_token].item() if no_token else 0
        maybe_prob = probs[0, maybe_token].item() if maybe_token else 0

        if yes_prob > no_prob and yes_prob > maybe_prob:
            return 'entailment'
        elif no_prob > yes_prob and no_prob > maybe_prob:
            return 'contradiction'
        else:
            return 'neutral'


@torch.no_grad()
def evaluate_snli(model, tokenizer, device, max_samples=None):
    """Evaluate model on SNLI test set."""
    # Load SNLI dataset
    dataset = load_dataset("snli", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0
    label_counts = {'entailment': 0, 'contradiction': 0, 'neutral': 0}
    correct_counts = {'entailment': 0, 'contradiction': 0, 'neutral': 0}

    for example in tqdm(dataset, desc="Evaluating SNLI"):
        premise = example['premise']
        hypothesis = example['hypothesis']
        true_label = example['label']

        # Skip invalid labels
        if true_label == -1:
            continue

        label_str = ['entailment', 'neutral', 'contradiction'][true_label]
        label_counts[label_str] += 1

        # Create prompt
        prompt = create_prompt(premise, hypothesis)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=model.max_seq_len)

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate prediction
        outputs = model(**inputs)
        pred_label = get_prediction(outputs.logits, tokenizer)

        # Check if correct
        if pred_label == label_str:
            correct += 1
            correct_counts[label_str] += 1

        total += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0

    # Per-class accuracy
    entail_acc = correct_counts['entailment'] / label_counts['entailment'] if label_counts['entailment'] > 0 else 0
    neutral_acc = correct_counts['neutral'] / label_counts['neutral'] if label_counts['neutral'] > 0 else 0
    contradict_acc = correct_counts['contradiction'] / label_counts['contradiction'] if label_counts['contradiction'] > 0 else 0

    results = {
        'accuracy': accuracy,
        'total_samples': total,
        'entailment_accuracy': entail_acc,
        'neutral_accuracy': neutral_acc,
        'contradiction_accuracy': contradict_acc,
        'label_distribution': label_counts,
        'correct_distribution': correct_counts
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on SNLI")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["gpt2", "grassmann"], required=True)
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--model_dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize W&B
    wandb.init(
        project="grassmann-flows-snli-eval",
        name=f"{args.model_type}_snli_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_type": args.model_type,
            "model_path": args.model_path,
            "max_samples": args.max_samples,
        },
    )

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer)

    # Load model
    print(f"Loading {args.model_type} model from {args.model_path}...")
    model = load_model(
        args.model_path,
        args.model_type,
        vocab_size,
        args.max_seq_len,
        args.model_dim,
        args.num_layers
    )
    model.to(device)

    # Evaluate
    print("Evaluating on SNLI...")
    results = evaluate_snli(model, tokenizer, device, args.max_samples)

    # Print results
    print("\nSNLI Evaluation Results:")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Entailment Accuracy: {results['entailment_accuracy']:.4f}")
    print(f"Neutral Accuracy: {results['neutral_accuracy']:.4f}")
    print(f"Contradiction Accuracy: {results['contradiction_accuracy']:.4f}")

    # Log to W&B
    wandb.log({
        "snli/accuracy": results['accuracy'],
        "snli/total_samples": results['total_samples'],
        "snli/entailment_accuracy": results['entailment_accuracy'],
        "snli/neutral_accuracy": results['neutral_accuracy'],
        "snli/contradiction_accuracy": results['contradiction_accuracy'],
    })

    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    # Finish W&B
    wandb.finish()


if __name__ == "__main__":
    main()