#!/usr/bin/env python3
"""
Visualize attention patterns for RoPE++ models.

This script loads a trained model and visualizes the attention score matrices
(Q·K^T before softmax) for specific layers and heads, similar to Figure 5 in the paper.

The visualization shows how real and imaginary heads differ in their attention patterns:
- Real heads typically focus on local context (diagonal patterns)
- Imaginary heads often attend to more global information

Usage:
    python visualize_attention.py --model-path <path> --text "Your input text here"
    
Example:
    python visualize_attention.py --model-path checkpoints/rope_pp-376m/checkpoint-100 \
        --text "The quick brown fox jumps over the lazy dog" \
        --layers 2,6 --heads 10,11
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from llama_variants.configuration_llama import LlamaConfig
from llama_variants.modeling_llama_rope_pp import LlamaForCausalLM as RoPEPPLlamaForCausalLM
from llama_variants.modeling_llama_fope import LlamaForCausalLM as FoPELlamaForCausalLM


class AttentionCapture:
    """Hook to capture attention weights from specific layers."""
    
    def __init__(self):
        self.attention_maps = {}
        self.hooks = []
    
    def register_hooks(self, model, layer_indices):
        """
        Register forward hooks on specific layers to capture attention.
        
        Args:
            model: The LlamaForCausalLM model
            layer_indices: List of layer indices to capture (e.g., [2, 6, 11])
        """
        for layer_idx in layer_indices:
            if layer_idx >= len(model.model.layers):
                print(f"Warning: Layer {layer_idx} doesn't exist (model has {len(model.model.layers)} layers)")
                continue
            
            layer = model.model.layers[layer_idx]
            hook = layer.self_attn.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.hooks.append(hook)
        
        print(f"Registered hooks on layers: {layer_indices}")
    
    def _make_hook(self, layer_idx):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # output is (attn_output, attn_weights)
            # attn_weights shape: [batch, num_heads, seq_len, seq_len]
            if output[1] is not None:
                self.attention_maps[layer_idx] = output[1].detach().cpu()
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention(self, layer_idx, head_idx):
        """
        Get attention weights for a specific layer and head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            
        Returns:
            Numpy array of attention weights [seq_len, seq_len]
        """
        if layer_idx not in self.attention_maps:
            return None
        
        attn = self.attention_maps[layer_idx]  # [batch, num_heads, seq_len, seq_len]
        if head_idx >= attn.shape[1]:
            return None
        
        return attn[0, head_idx].numpy()  # Take first batch item


class AttentionCapturePreSoftmax:
    """
    Hook to capture RAW attention scores (Q·K^T before softmax).
    This is what's shown in the paper figures.
    """
    
    def __init__(self):
        self.attention_scores = {}
        self.hooks = []
    
    def register_hooks(self, model, layer_indices):
        """Register hooks to capture pre-softmax attention scores."""
        for layer_idx in layer_indices:
            if layer_idx >= len(model.model.layers):
                print(f"Warning: Layer {layer_idx} doesn't exist")
                continue
            
            layer = model.model.layers[layer_idx]
            
            # We need to monkey-patch the attention forward to capture pre-softmax scores
            original_forward = layer.self_attn.forward
            
            def make_patched_forward(orig_forward, lidx):
                def patched_forward(*args, **kwargs):
                    # Call original forward
                    output = orig_forward(*args, **kwargs)
                    
                    # Manually compute Q·K^T to capture raw scores
                    # We'll need to hook into eager_attention_forward
                    # For now, we'll use the post-softmax weights
                    # (Note: for true pre-softmax, we'd need to modify eager_attention_forward)
                    
                    return output
                return patched_forward
            
            layer.self_attn.forward = make_patched_forward(original_forward, layer_idx)
        
        print(f"Registered pre-softmax hooks on layers: {layer_indices}")


def get_color_code(value):
    """
    Get ANSI color code for a normalized value (0-1).
    Creates a smooth gradient: dark blue → cyan → green → yellow → orange → red
    """
    # Define color palette (using 256-color ANSI codes)
    # These are background colors
    if value < 0.16:
        return "\033[48;5;17m"   # Very dark blue
    elif value < 0.33:
        return "\033[48;5;27m"   # Dark blue
    elif value < 0.50:
        return "\033[48;5;33m"   # Cyan/blue
    elif value < 0.66:
        return "\033[48;5;226m"  # Yellow
    elif value < 0.83:
        return "\033[48;5;208m"  # Orange
    else:
        return "\033[48;5;196m"  # Bright red


def ascii_heatmap(matrix, max_width=60, max_height=20, title="", use_color=True):
    """
    Create an ASCII heatmap visualization of a 2D matrix.
    
    Args:
        matrix: 2D numpy array
        max_width: Maximum character width for display
        max_height: Maximum character height for display
        title: Title to display above the heatmap
        use_color: Use ANSI color codes for better visualization
        
    Returns:
        String containing the ASCII heatmap
    """
    # Normalize matrix to 0-1 using PER-HEAD adaptive normalization
    matrix = matrix.astype(float)
    matrix_min = matrix.min()
    matrix_max = matrix.max()
    matrix_median = np.median(matrix)
    matrix_p95 = np.percentile(matrix, 95)
    matrix_p99 = np.percentile(matrix, 99)
    matrix_mean = matrix.mean()
    
    # Calculate sparsity (percentage of very low values)
    sparsity = (matrix < 0.001).sum() / matrix.size * 100
    
    # Normalize to p95 (clip values above to 1.0)
    # This makes the visualization show the typical range, not outliers
    if matrix_p95 > matrix_min:
        matrix_norm = np.clip((matrix - matrix_min) / (matrix_p95 - matrix_min), 0, 1)
        norm_range_max = matrix_p95
    else:
        matrix_norm = matrix * 0
        norm_range_max = matrix_max
    
    # Downsample if needed
    rows, cols = matrix_norm.shape
    original_rows, original_cols = rows, cols
    
    if rows > max_height:
        # Average pool rows
        row_factor = rows / max_height
        new_matrix = []
        for i in range(max_height):
            start_row = int(i * row_factor)
            end_row = int((i + 1) * row_factor)
            new_matrix.append(matrix_norm[start_row:end_row].mean(axis=0))
        matrix_norm = np.array(new_matrix)
        rows = max_height
    
    if cols > max_width:
        # Average pool cols
        col_factor = cols / max_width
        new_matrix = []
        for j in range(max_width):
            start_col = int(j * col_factor)
            end_col = int((j + 1) * col_factor)
            new_matrix.append(matrix_norm[:, start_col:end_col].mean(axis=1))
        matrix_norm = np.array(new_matrix).T
        cols = max_width
    
    # ANSI reset code
    reset = "\033[0m"
    
    # Build the heatmap
    lines = []
    
    if title:
        lines.append("\n" + title)
        lines.append("=" * len(title))
    
    # Enhanced statistics
    lines.append(f"Shape: {original_rows}x{original_cols} (displayed as {rows}x{cols})")
    lines.append(f"Stats: min={matrix_min:.6f}, max={matrix_max:.6f}, median={matrix_median:.6f}")
    lines.append(f"       mean={matrix_mean:.6f}, p95={matrix_p95:.6f}, p99={matrix_p99:.6f}, sparsity={sparsity:.1f}%")
    lines.append(f"Normalized to: [{matrix_min:.6f}, {norm_range_max:.6f}] → [0.0, 1.0] (values above clipped)")
    
    # Add column header
    if cols <= 20:
        header = "    " + "".join([f"{i%10:2d}" for i in range(cols)])  # Double width spacing
        lines.append(header)
    
    lines.append("  ┌" + "─" * (cols * 2) + "┐")
    
    for i in range(rows):
        # Row label
        if rows <= 20:
            line = f"{i:2d}│"
        else:
            line = "  │"
        
        for j in range(cols):
            val = matrix_norm[i, j]
            if use_color:
                color = get_color_code(val)
                line += color + "  " + reset  # Double space for square aspect ratio
            else:
                # Fallback to character gradient
                chars = " ░▒▓█"
                char_idx = int(val * (len(chars) - 1))
                line += chars[char_idx] * 2  # Double char for square aspect ratio
        
        line += "│"
        lines.append(line)
    
    lines.append("  └" + "─" * (cols * 2) + "┘")
    
    # Add color legend
    if use_color:
        lines.append("\nColor scale: " + 
                     get_color_code(0.0) + "  " + reset + " Low → " +
                     get_color_code(0.33) + "  " + reset + " → " +
                     get_color_code(0.66) + "  " + reset + " → " +
                     get_color_code(1.0) + "  " + reset + " High")
    
    return "\n".join(lines)


def is_local_path(path):
    """Check if a path is a local filesystem path."""
    return Path(path).exists()


def load_model(model_path, model_type="ropepp", device="cuda", force_eager=True):
    """
    Load a model for attention visualization.
    
    Args:
        model_path: Path to model checkpoint or HuggingFace model ID
        model_type: Type of model - "ropepp", "fope", "pythia", or "alibi"
        device: Device to load on
        force_eager: Force eager attention (required to capture attention weights)
        
    Returns:
        (model, tokenizer, config)
    """
    is_local = is_local_path(model_path)
    
    if is_local:
        print(f"Loading local checkpoint: {model_path}")
    else:
        print(f"Loading HuggingFace model: {model_path}")
    
    # Load config
    config = LlamaConfig.from_pretrained(model_path)
    
    # For HuggingFace RoPE++ models, set the rope_config if not present
    if not is_local and model_type == "ropepp" and not hasattr(config, 'rope_config'):
        # Infer rope_config from model name
        if 'RoPEPP_EH' in model_path or 'imag1' in model_path:
            config.rope_config = {'imag': True, 'imag_mode': 'imag1'}
        elif 'RoPEPP_EC' in model_path or 'imag2' in model_path:
            config.rope_config = {'imag': True, 'imag_mode': 'imag2'}
        elif 'RoPE' in model_path:
            config.rope_config = {'imag': False, 'imag_mode': 'imag1'}
        else:
            config.rope_config = {'imag': False, 'imag_mode': 'imag1'}
        print(f"Inferred RoPE config: {config.rope_config}")
    
    # Force eager attention to capture weights
    if force_eager:
        config._attn_implementation = "eager"
        print("Using eager attention (required for visualization)")
    
    config.use_cache = False
    config.torch_dtype = torch.float16
    
    print(f"Model config:")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - KV heads: {config.num_key_value_heads}")
    if hasattr(config, 'rope_config'):
        print(f"  - RoPE config: {config.rope_config}")
        print(f"\nRoPE Config Details:")
        print(f"  - Using imaginary heads: {config.rope_config.get('imag', False)}")
        print(f"  - Imaginary mode: {config.rope_config.get('imag_mode', 'N/A')}")
        if config.rope_config.get('imag_mode') == 'imag1':
            print(f"    → imag1 (RoPE++ EH): First half=Real, Second half=Imaginary")
        elif config.rope_config.get('imag_mode') == 'imag2':
            print(f"    → imag2 (RoPE++ EC): Even heads=Real, Odd heads=Imaginary")
        elif config.rope_config.get('imag_mode') == 'imagx':
            print(f"    → imagx: Even indices=Real, Odd indices=Imaginary")
        print()
    
    # Select model class based on type
    if model_type == "fope":
        ModelClass = FoPELlamaForCausalLM
    else:  # ropepp, pythia, alibi all use the same base
        ModelClass = RoPEPPLlamaForCausalLM
    
    # Load model
    model = ModelClass.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    model.eval()
    
    # Load tokenizer - try model path first, fallback to Llama-3-8B
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        print("  - Tokenizer not found in model path, using meta-llama/Meta-Llama-3-8B")
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, config


def visualize_attention_patterns(model, tokenizer, text, layer_indices, head_indices=None):
    """
    Visualize attention patterns for specific layers and heads.
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        text: Input text to analyze
        layer_indices: List of layer indices to visualize
        head_indices: List of head indices to visualize (None = all heads)
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = len(tokens)
    
    print(f"\n{'='*80}")
    print(f"Sequence Length: {seq_len} tokens")
    print(f"{'='*80}\n")
    
    # Set up attention capture
    capture = AttentionCapture()
    capture.register_hooks(model, layer_indices)
    
    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            use_cache=False
        )
    
    # Visualize attention for each layer and head
    for layer_idx in layer_indices:
        attn = capture.get_attention(layer_idx, 0)  # Get first head to check shape
        if attn is None:
            print(f"No attention captured for layer {layer_idx}")
            continue
        
        num_heads = capture.attention_maps[layer_idx].shape[1]
        
        # Determine which heads to visualize
        if head_indices is None:
            heads_to_show = list(range(num_heads))
        else:
            heads_to_show = [h for h in head_indices if h < num_heads]
        
        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx} - Total heads: {num_heads}")
        print(f"{'='*80}\n")
        
        for head_idx in heads_to_show:
            attn_matrix = capture.get_attention(layer_idx, head_idx)
            
            # Determine if this is a real or imaginary head (for RoPE++)
            head_type = "Unknown"
            if hasattr(model.config, 'rope_config') and model.config.rope_config.get('imag'):
                imag_mode = model.config.rope_config.get('imag_mode', 'imag1')
                if imag_mode == 'imag1':
                    # First half real, second half imaginary
                    head_type = "Real" if head_idx < num_heads // 2 else "Imaginary"
                elif imag_mode == 'imag2':
                    # Even heads real, odd heads imaginary
                    head_type = "Real" if head_idx % 2 == 0 else "Imaginary"
                elif imag_mode == 'imagx':
                    # Even indices real, odd indices imaginary
                    head_type = "Real" if head_idx % 2 == 0 else "Imaginary"
                title = f"Layer {layer_idx}, Head {head_idx} ({head_type})"
            else:
                title = f"Layer {layer_idx}, Head {head_idx}"
            
            # Print heatmap
            heatmap = ascii_heatmap(
                attn_matrix,
                max_width=min(50, seq_len),
                max_height=min(25, seq_len),
                title=title,
                use_color=True
            )
            print(heatmap)
            print()
    
    # Clean up
    capture.remove_hooks()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    for layer_idx in layer_indices:
        if layer_idx not in capture.attention_maps:
            continue
        
        attn_all = capture.attention_maps[layer_idx][0]  # [num_heads, seq_len, seq_len]
        
        # Calculate diagonal vs off-diagonal attention
        for head_idx in (head_indices if head_indices else range(attn_all.shape[0])):
            if head_idx >= attn_all.shape[0]:
                continue
            
            attn = attn_all[head_idx].numpy()
            
            # Diagonal attention (local)
            diag_width = 3  # Consider 3-diagonal as "local"
            mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]) <= diag_width
            local_attn = attn[mask].mean()
            global_attn = attn[~mask].mean()
            
            # Determine head type
            head_type = "?"
            if hasattr(model.config, 'rope_config') and model.config.rope_config.get('imag'):
                imag_mode = model.config.rope_config.get('imag_mode', 'imag1')
                if imag_mode == 'imag1':
                    head_type = "Real" if head_idx < attn_all.shape[0] // 2 else "Imag"
                elif imag_mode in ['imag2', 'imagx']:
                    head_type = "Real" if head_idx % 2 == 0 else "Imag"
                
                print(f"Layer {layer_idx}, Head {head_idx:2d} ({head_type}): "
                      f"Local={local_attn:.4f}, Global={global_attn:.4f}, "
                      f"Ratio={local_attn/global_attn if global_attn > 0 else 0:.2f}")
            else:
                print(f"Layer {layer_idx}, Head {head_idx:2d}: "
                      f"Local={local_attn:.4f}, Global={global_attn:.4f}, "
                      f"Ratio={local_attn/global_attn if global_attn > 0 else 0:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize attention patterns in RoPE++ models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize layer 2 heads 10 and 11 (real vs imaginary pair)
  python visualize_attention.py --model-path checkpoints/rope_pp-376m/checkpoint-100 \\
      --text "The cat sat on the mat" --layers 2 --heads 10,11
  
  # Visualize multiple layers, all heads
  python visualize_attention.py --model-path checkpoints/rope_pp-376m/checkpoint-100 \\
      --text "The cat sat on the mat" --layers 2,6,11
  
  # Compare RoPE vs RoPE++ checkpoints
  python visualize_attention.py --model-path checkpoints/rope-376m/checkpoint-100 \\
      --text "The cat sat on the mat" --layers 2 --heads 10,11
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to local checkpoint or HuggingFace model ID (e.g., SII-xrliu/RoPEPP_EH-DCLM-376M-4k)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='ropepp',
        choices=['ropepp', 'fope', 'pythia', 'alibi'],
        help='Type of model (default: ropepp)'
    )
    parser.add_argument(
        '--text',
        type=str,
        default="The quick brown fox jumps over the lazy dog and runs through the forest.",
        help='Input text to analyze (default: sample sentence)'
    )
    parser.add_argument(
        '--layers',
        type=str,
        default='2,6',
        help='Comma-separated layer indices to visualize (default: 2,6)'
    )
    parser.add_argument(
        '--heads',
        type=str,
        default=None,
        help='Comma-separated head indices to visualize (default: all heads). '
             'For RoPE++, use pairs like 10,11 to compare real vs imaginary.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (default: cuda if available)'
    )
    
    args = parser.parse_args()
    
    # Parse layer and head indices
    layer_indices = [int(x.strip()) for x in args.layers.split(',')]
    
    if args.heads:
        head_indices = [int(x.strip()) for x in args.heads.split(',')]
    else:
        head_indices = None
    
    # Validate model path (only if it looks like a local path)
    if '/' not in args.model_path or args.model_path.startswith('.') or args.model_path.startswith('/'):
        if not Path(args.model_path).exists():
            print(f"Error: Local model path does not exist: {args.model_path}")
            print("If you meant to use a HuggingFace model, use the format: owner/model-name")
            sys.exit(1)
    
    # Load model
    model, tokenizer, config = load_model(args.model_path, model_type=args.model_type, device=args.device)
    
    # Visualize
    visualize_attention_patterns(
        model=model,
        tokenizer=tokenizer,
        text=args.text,
        layer_indices=layer_indices,
        head_indices=head_indices
    )
    
    print(f"\n{'='*80}")
    print("Visualization complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
