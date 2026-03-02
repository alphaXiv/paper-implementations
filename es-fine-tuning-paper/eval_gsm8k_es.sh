#!/bin/bash
set -x

# Evaluation script for ES Fine-tuned GSM8K model using vLLM

# Usage:
# ./eval_gsm8k_es.sh <trained_model_path>
# Example: ./eval_gsm8k_es.sh es-ft-gsm8k-experiment/gsm8k_nccl_20260227_103714/model_saves/final_model_iteration_1000/pytorch_model.pth

if [ -z "$1" ]; then
    echo "Error: Please provide the trained model path"
    echo "Usage: $0 <trained_model_path>"
    echo "Example: $0 es-ft-gsm8k-experiment/gsm8k_nccl_20260227_103714/model_saves/final_model_iteration_1000/pytorch_model.pth"
    exit 1
fi

TRAINED_MODEL_PATH="$1"

# Check if the trained model exists
if [ ! -f "$TRAINED_MODEL_PATH" ]; then
    echo "Error: Trained model not found at $TRAINED_MODEL_PATH"
    exit 1
fi

echo "Evaluating trained model from: $TRAINED_MODEL_PATH"

# Default configuration
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-src/data/gsm8k-0.1/test.parquet}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-src/evals/qwen2.5_3b_instruct_eval_results_gsm8k_es_0.1_n_30}"

echo "Configuration:"
echo "  Base Model: $MODEL_ID"
echo "  Eval Data: $EVAL_DATA_PATH"
echo "  Eval Samples: $EVAL_SAMPLES"
echo "  Max New Tokens: $MAX_NEW_TOKENS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# Run evaluation
python3 eval_gsm8k_vllm.py \
    --model_id "$MODEL_ID" \
    --trained_model_path "$TRAINED_MODEL_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --eval_samples $EVAL_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --batch_size $BATCH_SIZE \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --output_dir "$OUTPUT_DIR" \
    --save_responses \
    --show_examples 10 \
    --verbose

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Evaluation completed successfully! ==="
    echo "Results saved to: $OUTPUT_DIR"
else
    echo ""
    echo "=== Evaluation failed! ==="
    exit 1
fi
