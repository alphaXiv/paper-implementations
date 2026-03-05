#!/bin/bash
set -e

# Script to prepare custom tokenizers for base models (Llama and Qwen)
# These tokenizers add the simple chat template: "Question: {input} Answer: Let's think step by step."

echo "=========================================="
echo "Preparing Base Model Tokenizers"
echo "=========================================="

# Create tokenizers directory
mkdir -p ./tokenizers

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set"
    echo "You may need to login to HuggingFace to access Llama models"
    echo "Run: export HF_TOKEN=your_huggingface_token"
fi

# Prepare Llama 3.2 3B base model tokenizer
echo ""
echo "Creating tokenizer for Llama-3.2-3B (base)..."
python3 base_model_tokenizer.py \
    --model_path meta-llama/Llama-3.2-3B \
    --save_path ./tokenizers/llama-3.2-3b-base-chat \
    --test

echo ""
echo "✓ Llama tokenizer created at: ./tokenizers/llama-3.2-3b-base-chat"

# Prepare Qwen2.5 3B base model tokenizer
echo ""
echo "Creating tokenizer for Qwen2.5-3B (base)..."
python3 base_model_tokenizer.py \
    --model_path Qwen/Qwen2.5-3B \
    --save_path ./tokenizers/qwen2.5-3b-base-chat \
    --test

echo ""
echo "✓ Qwen tokenizer created at: ./tokenizers/qwen2.5-3b-base-chat"

echo ""
echo "=========================================="
echo "Tokenizer Preparation Complete!"
echo "=========================================="
echo "Use these tokenizer paths in your training config:"
echo "  Llama: ./tokenizers/llama-3.2-3b-base-chat"
echo "  Qwen:  ./tokenizers/qwen2.5-3b-base-chat"
