#!/bin/bash

# Attention Pattern Visualizer for RoPE++ Models
# This script visualizes attention patterns from trained checkpoints

# Install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv --python 3.12

# Activate venv first
source .venv/bin/activate

# Install all dependencies
uv sync --extra gpu

echo "=========================================="
echo "RoPE++ Attention Pattern Visualizer"
echo "=========================================="
echo ""

# Default parameters (can be overridden by user)
TEXT="${TEXT:-The quick brown fox jumps over the lazy dog and runs through the forest.}"
LAYERS="${LAYERS:-2,6}"
HEADS="${HEADS:-10,11}"

# If MODEL_PATH is not set, use default HuggingFace model
if [ -z "$MODEL_PATH" ]; then
    # Default model: RoPEPP_EC-DCLM-376M-32k
    MODEL_PATH="SII-xrliu/RoPEPP_EC-DCLM-376M-32k"
    MODEL_TYPE="ropepp"
    
    echo "Using default model: $MODEL_PATH"
    echo "(To use a different model: export MODEL_PATH='SII-xrliu/model-name' before running)"
    echo ""
else
    # Check if it's a local path or HF model
    if [ -d "$MODEL_PATH" ]; then
        echo "Using local checkpoint: $MODEL_PATH"
        MODEL_TYPE="${MODEL_TYPE:-ropepp}"
    else
        echo "Using HuggingFace model: $MODEL_PATH"
        MODEL_TYPE="${MODEL_TYPE:-ropepp}"
    fi
fi

echo ""
echo "Model Path: $MODEL_PATH"
echo "Model Type: $MODEL_TYPE"
echo "Input Text: $TEXT"
echo "Layers to visualize: $LAYERS"
echo "Heads to visualize: $HEADS"
echo ""
echo "=========================================="
echo "Running Visualization..."
echo "=========================================="
echo ""

# Check if sample text file exists, otherwise use default
if [ -f "visualizer/sample-wiki-point.txt" ]; then
    # Load full text and truncate to ~10,000 tokens (~50KB of text)
    TEXT=$(head -c 50000 visualizer/sample-wiki-point.txt)
    echo "Using text from visualizer/sample-wiki-point.txt (truncated to ~10,000 tokens for memory)"
else
    TEXT="${TEXT:-The quick brown fox jumps over the lazy dog and runs through the forest.}"
    echo "Using default text"
fi

# Run the visualization
python visualizer/visualize_attention.py \
    --model-path "$MODEL_PATH" \
    --model-type "$MODEL_TYPE" \
    --text "$TEXT" \
    --layers "$LAYERS" \
    --heads "$HEADS"

echo ""
echo "=========================================="
echo "Visualization Complete!"
echo "=========================================="
echo ""
echo "To customize the visualization, you can:"
echo "  export MODEL_PATH='SII-xrliu/RoPEPP_EH-DCLM-376M-4k'  # HF model"
echo "  export MODEL_PATH='checkpoints/my-model/checkpoint-100'  # Local checkpoint"
echo "  export MODEL_TYPE='ropepp'  # or 'fope', 'pythia', 'alibi'"
echo "  export TEXT='Your custom input text here'"
echo "  export LAYERS='2,6,11'  # Layer indices to visualize"
echo "  export HEADS='10,11'    # Head indices (for RoPE++: pairs like 10,11 are real/imag)"
echo ""
echo "Or run directly:"
echo "  python visualizer/visualize_attention.py --model-path SII-xrliu/RoPEPP_EH-DCLM-376M-4k \\"
echo "      --text 'The cat sat on the mat' --layers 2,6 --heads 10,11"
echo ""
