#!/bin/bash
set -e

# Countdown Model Evaluation Script
# This script merges FSDP checkpoints and evaluates the fine-tuned model on the test set

echo "=========================================="
echo "Countdown Model Evaluation Pipeline"
echo "=========================================="

# Configuration
PROJECT_NAME="verl_grpo_countdown_base_custom"
EXPERIMENT_NAME="llama3.2_3b_base_custom_template_lora"
# BASE_MODEL="Qwen/Qwen2.5-3B"
BASE_MODEL='meta-llama/Llama-3.2-3b'

TEST_FILE="./data/countdown-0.1/test.parquet"
TASK_TYPE="countdown"

# Parse command line arguments
CHECKPOINT_STEP=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --project_name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--step CHECKPOINT_STEP] [--base_model MODEL_PATH] [--project_name NAME] [--experiment_name NAME]"
            exit 1
            ;;
    esac
done

# Auto-detect latest checkpoint if not specified
if [ -z "$CHECKPOINT_STEP" ]; then
    echo "Auto-detecting latest checkpoint..."
    CHECKPOINT_BASE="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
    if [ ! -d "$CHECKPOINT_BASE" ]; then
        echo "Error: Checkpoint directory not found: $CHECKPOINT_BASE"
        exit 1
    fi
    
    # Find the latest global_step directory
    LATEST_STEP=$(ls -d ${CHECKPOINT_BASE}/global_step_* 2>/dev/null | sort -V | tail -1)
    if [ -z "$LATEST_STEP" ]; then
        echo "Error: No checkpoint found in $CHECKPOINT_BASE"
        exit 1
    fi
    CHECKPOINT_STEP=$(basename "$LATEST_STEP" | sed 's/global_step_//')
    echo "Found latest checkpoint: step $CHECKPOINT_STEP"
fi

# Set paths
CHECKPOINT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${CHECKPOINT_STEP}"
ACTOR_DIR="${CHECKPOINT_DIR}/actor"
MERGED_DIR="${CHECKPOINT_DIR}/merged_model"
OUTPUT_FILE="evals/llama3.2_3b_base_eval_results_countdown_${CHECKPOINT_STEP}.json"

echo ""
echo "Configuration:"
echo "  Project: $PROJECT_NAME"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Checkpoint Step: $CHECKPOINT_STEP"
echo "  Base Model: $BASE_MODEL"
echo "  Test File: $TEST_FILE"
echo "  Output File: $OUTPUT_FILE"
echo ""

# Create evals directory if it doesn't exist
mkdir -p evals

# Check if checkpoint exists
if [ ! -d "$ACTOR_DIR" ]; then
    echo "Error: Checkpoint directory not found: $ACTOR_DIR"
    exit 1
fi

# Step 1: Merge FSDP checkpoint shards
echo "=========================================="
echo "Step 1: Merging FSDP checkpoint shards"
echo "=========================================="

if [ -d "$MERGED_DIR" ] && [ "$(ls -A $MERGED_DIR)" ]; then
    echo "Merged model already exists at: $MERGED_DIR"
    read -p "Do you want to re-merge? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping merge step..."
    else
        echo "Removing existing merged model..."
        rm -rf "$MERGED_DIR"
        mkdir -p "$MERGED_DIR"
        
        echo "Merging checkpoint..."
        python3 verl/scripts/model_merger.py \
            --backend fsdp \
            --hf_model_path "$BASE_MODEL" \
            --local_dir "$ACTOR_DIR" \
            --target_dir "$MERGED_DIR"
        
        echo "✓ Merge completed successfully!"
    fi
else
    mkdir -p "$MERGED_DIR"
    
    echo "Merging checkpoint..."
    python3 verl/scripts/model_merger.py \
        --backend fsdp \
        --hf_model_path "$BASE_MODEL" \
        --local_dir "$ACTOR_DIR" \
        --target_dir "$MERGED_DIR"
    
    echo "✓ Merge completed successfully!"
fi

echo ""

# Step 2: Run evaluation
echo "=========================================="
echo "Step 2: Evaluating on test set"
echo "=========================================="

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found: $TEST_FILE"
    echo "Please ensure countdown data is prepared."
    exit 1
fi

echo "Running evaluation..."
echo "This may take several minutes depending on GPU speed..."
echo ""

python3 evaluate_model.py \
    --model_path "$MERGED_DIR" \
    --test_file "$TEST_FILE" \
    --task_type "$TASK_TYPE" \
    --output_file "$OUTPUT_FILE"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Display results if jq is available
if command -v jq &> /dev/null; then
    echo "Quick Summary:"
    jq -r '"Total: \(.total) | Correct: \(.correct) | Accuracy: \(.accuracy * 100 | floor)%"' "$OUTPUT_FILE"
else
    echo "Install 'jq' to see a quick summary here."
    echo "Otherwise, check the full results in: $OUTPUT_FILE"
fi
