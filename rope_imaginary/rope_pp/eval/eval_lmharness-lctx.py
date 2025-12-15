#!/usr/bin/env python3
"""
Evaluation script for RoPE++ and FoPE models using lm-evaluation-harness.

Supports:
- RoPE++ models (with various rope configurations)
- FoPE models (Functional Positional Encoding)
- Pythia models (learned positional embeddings)
- ALiBi models (Attention with Linear Biases)
- Local checkpoints from training

Tasks evaluated:
- BABILong: Long-context reasoning benchmark
"""

import sys
import os
import torch
import json
import argparse
from pathlib import Path

from llama_variants.configuration_llama import LlamaConfig
from llama_variants.modeling_llama_rope_pp import LlamaForCausalLM as RoPEPPLlamaForCausalLM
from llama_variants.modeling_llama_fope import LlamaForCausalLM as FoPELlamaForCausalLM
from llama_variants.modeling_llama_pythia import LlamaForCausalLM as PythiaLlamaForCausalLM
from llama_variants.modeling_llama_alibi import LlamaForCausalLM as AlibiLlamaForCausalLM
from transformers import AutoTokenizer

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


def is_local_path(path):
    """
    Check if a path is a local filesystem path.
    
    Args:
        path: Path string to check
    
    Returns:
        True if path exists locally, False otherwise
    """
    return Path(path).exists()


def load_ropepp_model(model_path, rope_config, device="cuda"):
    """
    Load a RoPE++ model with custom rope configuration.
    
    Args:
        model_path: HuggingFace model path or local path
        rope_config: Dictionary with rope configuration (for HF models only, ignored for local)
        device: Device to load model on
    
    Returns:
        Loaded model and tokenizer
    """
    print(f"Loading RoPE++ model: {model_path}")
    
    # Load config
    config = LlamaConfig.from_pretrained(model_path)
    config._attn_implementation = "flash_attention_2"
    
    # For local checkpoints, rope_config is already in config.json
    # For HuggingFace models, use the provided rope_config
    if not is_local_path(model_path) and rope_config:
        config.__setattr__("rope_config", rope_config)
        print(f"Using provided RoPE config: {rope_config}")
    elif hasattr(config, 'rope_config'):
        print(f"Using RoPE config from checkpoint: {config.rope_config}")
    
    # Apply scaling factor if present
    if hasattr(config, 'rope_config') and isinstance(config.rope_config, dict) and 'scaling_factor' in config.rope_config:
        config.rope_theta = config.rope_theta * config.rope_config['scaling_factor']
    
    # Load model
    model = RoPEPPLlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device
    )
    
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return model, tokenizer


def load_fope_model(model_path, rope_config=None, device="cuda"):
    """
    Load a FoPE (Functional Positional Encoding) model.
    
    Args:
        model_path: HuggingFace model path or local path
        rope_config: Optional dictionary with rope configuration (typically just scaling_factor)
        device: Device to load model on
    
    Returns:
        Loaded model and tokenizer
    """
    print(f"Loading FoPE model: {model_path}")
    if rope_config:
        print(f"Config: {rope_config}")
    
    # Load config
    config = LlamaConfig.from_pretrained(model_path)
    
    # FoPE only needs scaling_factor (if provided)
    if rope_config and 'scaling_factor' in rope_config:
        config.rope_theta = config.rope_theta * rope_config['scaling_factor']
    
    # Load model
    model = FoPELlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device
    )
    
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return model, tokenizer


def load_pythia_model(model_path, rope_config=None, device="cuda"):
    """
    Load a Pythia model (no rotary positional encoding).
    
    Args:
        model_path: HuggingFace model path or local path
        rope_config: Optional dictionary (unused for Pythia, but kept for consistency)
        device: Device to load model on
    
    Returns:
        Loaded model and tokenizer
    """
    print(f"Loading Pythia model: {model_path}")
    
    # Load config
    config = LlamaConfig.from_pretrained(model_path)
    
    # Load model
    model = PythiaLlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device
    )
    
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return model, tokenizer


def load_alibi_model(model_path, rope_config=None, device="cuda"):
    """
    Load an ALiBi (Attention with Linear Biases) model.
    
    Args:
        model_path: HuggingFace model path or local path
        rope_config: Optional dictionary (unused for ALiBi, but kept for consistency)
        device: Device to load model on
    
    Returns:
        Loaded model and tokenizer
    """
    print(f"Loading ALiBi model: {model_path}")
    
    # Load config
    config = LlamaConfig.from_pretrained(model_path)
    
    # Load model
    model = AlibiLlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device
    )
    
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return model, tokenizer


def evaluate_model(model_name, model_path, rope_config, model_type="ropepp", max_seq_len=32768, batch_size=1):
    """
    Evaluate a single model on BABILong benchmarks.
    
    Args:
        model_name: Name for logging/results
        model_path: HuggingFace model path
        rope_config: RoPE configuration dict
        model_type: Type of model - "ropepp", "fope", "pythia", or "alibi"
        max_seq_len: Maximum sequence length (default 32768 for long-context tasks)
        batch_size: Batch size for evaluation (default 1 for long sequences)
    
    Returns:
        Results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}\n")
    
    # Load the model using the appropriate loader
    if model_type == "fope":
        model, tokenizer = load_fope_model(model_path, rope_config)
    elif model_type == "ropepp":
        model, tokenizer = load_ropepp_model(model_path, rope_config)
    elif model_type == "pythia":
        model, tokenizer = load_pythia_model(model_path, rope_config)
    elif model_type == "alibi":
        model, tokenizer = load_alibi_model(model_path, rope_config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'ropepp', 'fope', 'pythia', or 'alibi'")
    
    # Wrap in HFLM for lm-eval
    lm_eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_seq_len,
    )
    
    # Run evaluation
    results = simple_evaluate(
        model=lm_eval_model,
        tasks=["babilong"],
        batch_size=batch_size,
        log_samples=False,
        verbosity="INFO",
        metadata={"pretrained": model_path, "tokenizer": model_path}
    )
    
    # Clean up
    del model
    del lm_eval_model
    torch.cuda.empty_cache()
    
    return results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate RoPE++ models and variants on long-context tasks using lm-evaluation-harness'
    )
    parser.add_argument(
        '--local-checkpoint',
        type=str,
        default=None,
        help='Path to local checkpoint directory to evaluate (e.g., checkpoints/model/checkpoint-3)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Display name for the local checkpoint (default: uses checkpoint path)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='ropepp',
        choices=['ropepp', 'fope', 'pythia', 'alibi'],
        help='Type of model for local checkpoint (default: ropepp)'
    )
    parser.add_argument(
        '--include-baselines',
        action='store_true',
        help='Also evaluate baseline HuggingFace models (default: only evaluate local checkpoint)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for evaluation (default: 64)'
    )
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=32768,
        help='Maximum sequence length (default: 32768 for long-context)'
    )
    
    args = parser.parse_args()
    
    # Define models to evaluate
    # Format: (name, path, rope_config, model_type, max_out_len)
    models = []
    
    # Add local checkpoint if provided
    if args.local_checkpoint:
        checkpoint_path = args.local_checkpoint
        
        # Validate checkpoint exists
        if not is_local_path(checkpoint_path):
            print(f"ERROR: Local checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        # Determine model name
        if args.model_name:
            model_name = args.model_name
        else:
            # Generate name from path
            model_name = f"Local-{Path(checkpoint_path).parent.name}-{Path(checkpoint_path).name}"
        
        # For local checkpoints, rope_config will be loaded from config.json automatically
        # So we pass an empty dict here
        models.append((
            model_name,
            checkpoint_path,
            {},  # rope_config loaded from checkpoint
            args.model_type,
            64  # max_out_len
        ))
        
        print(f"\n{'='*80}")
        print(f"Will evaluate local checkpoint: {checkpoint_path}")
        print(f"Model name: {model_name}")
        print(f"Model type: {args.model_type}")
        print(f"Config will be loaded from checkpoint's config.json")
        print(f"{'='*80}\n")
    
    # Add baseline models if requested
    if args.include_baselines or not args.local_checkpoint:
        # Evaluate a bunch of different embedding variants from: https://huggingface.co/collections/SII-xrliu/rope
        # Uncomment as you wish!
        baseline_models = [
            # ('RoPE-DCLM-376M-32k', 'SII-xrliu/RoPE-DCLM-376M-32k', {'imag': False, 'imag_mode': 'imag1'}, 'ropepp', 64), 
            # ('RoPEPP_EH-DCLM-376M-32k', 'SII-xrliu/RoPEPP_EH-DCLM-376M-32k', {'imag': True, 'imag_mode': 'imag1'}, 'ropepp', 64), 
            # ('RoPEPP_EC-DCLM-376M-32k', 'SII-xrliu/RoPEPP_EC-DCLM-376M-32k', {'imag': True, 'imag_mode': 'imag2'}, 'ropepp', 64), 

            # ('RoPE-DCLM-776M-32k', 'SII-xrliu/RoPE-DCLM-776M-32k', {'imag': False, 'imag_mode': 'imag1'}, 'ropepp', 64), 
            ('RoPEPP_EH-DCLM-776M-4k', 'SII-xrliu/RoPEPP_EH-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag1'}, 'ropepp', 64), 
            ('RoPEPP_EH-DCLM-776M-32k', 'SII-xrliu/RoPEPP_EH-DCLM-776M-32k', {'imag': True, 'imag_mode': 'imag1'}, 'ropepp', 64), 
            ('RoPEPP_EC-DCLM-776M-32k', 'SII-xrliu/RoPEPP_EC-DCLM-776M-32k', {'imag': True, 'imag_mode': 'imag2'}, 'ropepp', 64), 

            # ('RoPE-DCLM-1_5B-32k', 'SII-xrliu/RoPE-DCLM-1_5B-32k', {'imag': False, 'imag_mode': 'imag1'}, 'ropepp', 64), 
            ('RoPEPP_EH-DCLM-1_5B-32k', 'SII-xrliu/RoPEPP_EH-DCLM-1_5B-32k', {'imag': True, 'imag_mode': 'imag1'}, 'ropepp', 64), 
            ('RoPEPP_EC-DCLM-1_5B-32k', 'SII-xrliu/RoPEPP_EC-DCLM-1_5B-32k', {'imag': True, 'imag_mode': 'imag2'}, 'ropepp', 64), 
        ]
        models.extend(baseline_models)
    
    # Ensure we have at least one model to evaluate
    if not models:
        print("ERROR: No models to evaluate. Use --local-checkpoint or --include-baselines")
        sys.exit(1)
    
    all_results = {}
    
    for model_name, model_path, rope_config, model_type, max_out_len in models:
        try:
            results = evaluate_model(
                model_name=model_name,
                model_path=model_path,
                rope_config=rope_config,
                model_type=model_type,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size
            )
            
            # Store results
            all_results[model_name] = results['results']
            
            # Print results for this model
            print(f"\n{model_name} Results:")
            
            # BABILong results
            if 'babilong' in results['results']:
                babilong_results = results['results']['babilong']
                acc = babilong_results.get('acc,none', babilong_results.get('acc', 0))
                print(f"  BABILong - Accuracy: {acc:.4f}")
            
        except Exception as e:
            print(f"\nError evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<40} {'BABILong':<12}")
    print("-" * 80)
    
    for model_name, result in all_results.items():
        if "error" in result:
            print(f"{model_name:<40} ERROR: {result['error']}")
        else:
            # Extract metrics
            babilong_acc = result.get('babilong', {}).get('acc,none', result.get('babilong', {}).get('acc', 0))

            print(f"{model_name:<40} {babilong_acc:.4f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
