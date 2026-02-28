#!/bin/bash
set -x

# ES Fine-tuning Script for GSM8K
# This script runs Evolution Strategies fine-tuning on the GSM8K dataset

echo "Starting ES Fine-tuning for GSM8K..."

# Check if HF_TOKEN is set for gated models (e.g., Llama)
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set"
    echo "If using gated models (e.g., Llama), you may encounter authentication errors"
    echo "Set with: export HF_TOKEN=your_huggingface_token"
else
    echo "HF_TOKEN found, logging in to HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN" 2>&1 | grep -v "Token is valid"
    echo "✓ Logged in to HuggingFace"
fi
echo ""

# Default hyperparameters
SIGMA=${SIGMA:-0.001}
ALPHA=${ALPHA:-0.0005}
POPULATION_SIZE=${POPULATION_SIZE:-8}
NUM_ENGINES=${NUM_ENGINES:-8}
NUM_ITERATIONS=${NUM_ITERATIONS:-100}
NUM_TRAIN_SAMPLES=${NUM_TRAIN_SAMPLES:-200}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3,4,5,6,7"}
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-3B-Instruct"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"es-ft-gsm8k-experiment-exp3"}
DATA_PATH="src/data/gsm8k-0.1/train.parquet"
TOKENIZER_PATH=${TOKENIZER_PATH:-""}


echo "Configuration:"
echo "  Model: $MODEL_NAME"
if [ -n "$TOKENIZER_PATH" ]; then
    echo "  Tokenizer: $TOKENIZER_PATH (custom)"
else
    echo "  Tokenizer: Using model's default tokenizer"
fi
echo "  Sigma: $SIGMA"
echo "  Alpha: $ALPHA"
echo "  Population Size: $POPULATION_SIZE"
echo "  Number of Engines: $NUM_ENGINES"
echo "  Number of Iterations: $NUM_ITERATIONS"
echo "  Training Samples: $NUM_TRAIN_SAMPLES"
echo "  CUDA Devices: $CUDA_DEVICES"
echo "  Experiment Directory: $EXPERIMENT_DIR"
echo ""

# Check if data exists
if [ ! -f $DATA_PATH ]; then
    echo "Error: GSM8K training data not found at $DATA_PATH"
    echo "Please prepare the data first using grpo_data_gsm8k.py"
    exit 1
fi

# Run ES fine-tuning
CMD="python3 es_fine_tuning_gsm8k_accl.py \
    --model_name \"$MODEL_NAME\" \
    --sigma $SIGMA \
    --alpha $ALPHA \
    --population_size $POPULATION_SIZE \
    --num_engines $NUM_ENGINES \
    --num_iterations $NUM_ITERATIONS \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --cuda_devices \"$CUDA_DEVICES\" \
    --experiment_dir \"$EXPERIMENT_DIR\" \
    --data_path \"$DATA_PATH\" \
    --verbose"

# Add tokenizer path only if specified
if [ -n "$TOKENIZER_PATH" ]; then
    CMD="$CMD --tokenizer_path \"$TOKENIZER_PATH\""
fi

eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "ES Fine-tuning completed successfully!"
    echo "Model saved to: $EXPERIMENT_DIR/gsm8k_nccl_*/model_saves/final_model_iteration_$NUM_ITERATIONS"
else
    echo ""
    echo "ES Fine-tuning failed with exit code $?"
    exit 1
fi

# Optional: Evaluate the final model on test set
# Uncomment the following lines to run evaluation after training
# echo ""
# echo "Evaluating final model on test set..."
# python3 src/evaluate_model.py \
#     --model_path "$EXPERIMENT_DIR/gsm8k_nccl_*/model_saves/final_model_iteration_$NUM_ITERATIONS" \
#     --test_file src/data/gsm8k-0.1/test.parquet \
#     --task_type gsm8k \
#     --output_file gsm8k_es_eval_results.json
