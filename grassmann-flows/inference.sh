#!/bin/bash

set -e  # Exit on error

# ============================================================================
# Grassmann Flows - Inference & Generation Script
# ============================================================================
# This script provides inference capabilities for trained Grassmann Flow models.
#
# Usage: ./speedrun-inference.sh [MODEL_PATH] [PROMPT]
#   MODEL_PATH: Path to trained model checkpoint
#   PROMPT: Text prompt for generation (optional)
#
# Examples:
#   ./speedrun-inference.sh outputs/2024-01-01_12-00-00/grassmann_wikitext/
#   ./speedrun-inference.sh outputs/latest/ "The meaning of life is"
# ============================================================================

# Detect number of GPUs dynamically
if command -v nvidia-smi &> /dev/null; then
    DETECTED_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    DETECTED_GPUS=0
fi

# Configuration
MODEL_PATH=${1:-"outputs/latest"}
PROMPT=${2:-"The future of artificial intelligence"}

if [ "$DETECTED_GPUS" -gt 1 ]; then
    NUM_GPUS=$DETECTED_GPUS
else
    NUM_GPUS=1
fi

echo "=========================================="
echo "Grassmann Flows Inference"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo "Using GPUs: $NUM_GPUS"
echo "=========================================="
echo ""

# ============================================================================
# Step 0: Environment Setup
# ============================================================================

echo "[Step 0/2] Setting up environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install dependencies if needed
if ! python -c "import torch" &> /dev/null; then
    echo "Installing dependencies..."
    uv pip install -e .
fi

# ============================================================================
# Step 1: Create Inference Script
# ============================================================================

echo "[Step 1/2] Creating inference script..."

cat > inference_temp.py << 'EOF'
import torch
import argparse
from pathlib import Path
from transformers import GPT2Tokenizer
import sys
sys.path.insert(0, 'src')

from grassmann_flows.models import GPT2, GrassmannGPT

def load_model(model_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    model_path = Path(model_path)
    
    # Load config
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    import json
    with open(config_path) as f:
        config = json.load(f)
    
    # Determine model type
    model_type = config.get("model_type", "grassmann")
    
    # Create model
    if model_type == "gpt2":
        model = GPT2(
            vocab_size=config["vocab_size"],
            max_seq_len=config["max_seq_len"],
            model_dim=config["model_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            dropout=config["dropout"]
        )
    else:
        model = GrassmannGPT(
            vocab_size=config["vocab_size"],
            max_seq_len=config["max_seq_len"],
            model_dim=config["model_dim"],
            num_layers=config["num_layers"],
            reduced_dim=config.get("reduced_dim", 64),
            ff_dim=config["ff_dim"],
            dropout=config["dropout"],
            window_sizes=config.get("window_sizes", [1, 2, 4, 8, 16, 32])
        )
    
    # Load weights
    checkpoint_path = model_path / "model.pt"
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
    
    model.to(device)
    model.eval()
    return model, config

def generate_text(model, tokenizer, prompt: str, max_length: int = 100, temperature: float = 0.8):
    """Generate text from prompt."""
    device = next(model.parameters()).device
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits for next token
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, config = load_model(args.model_path, device)
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print("Generating...")
    
    generated_text = generate_text(
        model, tokenizer, args.prompt, 
        max_length=args.max_length, 
        temperature=args.temperature
    )
    
    print(f"\nGenerated text:\n{generated_text}")
    print(f"\nModel type: {config.get('model_type', 'unknown')}")
    print(f"Model dimension: {config.get('model_dim', 'unknown')}")

if __name__ == "__main__":
    main()
EOF

# ============================================================================
# Step 2: Run Inference
# ============================================================================

echo "[Step 2/2] Running inference..."

python inference_temp.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --max_length 100 \
    --temperature 0.8

# Clean up
rm inference_temp.py

echo ""
echo "=========================================="
echo "Inference complete!"
echo "=========================================="