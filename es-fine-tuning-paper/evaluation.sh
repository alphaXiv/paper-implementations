#!/bin/bash
# Evaluation runner for ES and GRPO trained models
# Runs evaluations on test sets and aggregates results

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
METHOD=""           # "es", "grpo", or "both"
TASK=""            # "gsm8k" or "countdown"
TRAIN_SPLIT=""     # Training split used (e.g., 0.1, 0.4)
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
MODEL_TYPE="instruct"
BATCH_SIZE=32
NUM_GPUS=4
CHECKPOINT_DIR=""
EVAL_SPLIT="test"
HF_TOKEN_VALUE=""

# Print usage
usage() {
    cat << EOF
${GREEN}Evaluation Runner for ES and GRPO Models${NC}
Usage: $0 --method <es|grpo|both> --task <gsm8k|countdown> --train-split <fraction> [OPTIONS]

${YELLOW}Required:${NC}
  --method METHOD          Evaluation method: 'es', 'grpo', or 'both'
  --task TASK              Task to evaluate: 'gsm8k' or 'countdown'
  --train-split FRACTION   Train split used during training (e.g., 0.1, 0.4)

${YELLOW}Optional:${NC}
  --model MODEL_NAME       Model name (default: Qwen/Qwen2.5-3B-Instruct)
  --model-type TYPE        Model type: 'instruct' or 'base' (default: instruct)
  --checkpoint-dir DIR     Override checkpoint directory auto-detection
  --batch-size N           Batch size for evaluation (default: 32)
  --num-gpus N             Number of GPUs to use (default: 4)
  --eval-split SPLIT       Split to evaluate on (default: test)
  --hf-token TOKEN         HuggingFace token (or set HF_TOKEN env var)

${YELLOW}Examples:${NC}
  # Evaluate ES model on GSM8K test set
  $0 --method es --task gsm8k --train-split 0.1

  # Evaluate GRPO model on Countdown
  $0 --method grpo --task countdown --train-split 0.4

  # Evaluate both methods
  $0 --method both --task gsm8k --train-split 0.1

  # Custom checkpoint directory
  $0 --method es --task gsm8k --train-split 0.1 --checkpoint-dir ./my_checkpoints

EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --train-split)
            TRAIN_SPLIT="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --eval-split)
            EVAL_SPLIT="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN_VALUE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$METHOD" ]; then
    echo -e "${RED}Error: --method is required${NC}"
    usage
fi

if [ "$METHOD" != "es" ] && [ "$METHOD" != "grpo" ] && [ "$METHOD" != "both" ]; then
    echo -e "${RED}Error: --method must be 'es', 'grpo', or 'both'${NC}"
    usage
fi

if [ -z "$TASK" ]; then
    echo -e "${RED}Error: --task is required${NC}"
    usage
fi

if [ "$TASK" != "gsm8k" ] && [ "$TASK" != "countdown" ]; then
    echo -e "${RED}Error: --task must be 'gsm8k' or 'countdown'${NC}"
    usage
fi

if [ -z "$TRAIN_SPLIT" ]; then
    echo -e "${RED}Error: --train-split is required${NC}"
    usage
fi

# Set environment variables
if [ -n "$HF_TOKEN_VALUE" ]; then
    export HF_TOKEN="$HF_TOKEN_VALUE"
fi

if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set. Some models may not be accessible.${NC}"
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set data directory
if [ "$TASK" == "gsm8k" ]; then
    DATA_DIR="./src/data/gsm8k-$TRAIN_SPLIT"
else
    DATA_DIR="./src/data/countdown-$TRAIN_SPLIT"
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Evaluation Runner                                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Method:       ${YELLOW}$METHOD${NC}"
echo -e "  Task:         ${YELLOW}$TASK${NC}"
echo -e "  Train Split:  ${YELLOW}$TRAIN_SPLIT${NC}"
echo -e "  Eval Split:   ${YELLOW}$EVAL_SPLIT${NC}"
echo -e "  Model:        ${YELLOW}$MODEL_NAME${NC}"
echo -e "  Model Type:   ${YELLOW}$MODEL_TYPE${NC}"
echo -e "  Batch Size:   ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  Num GPUs:     ${YELLOW}$NUM_GPUS${NC}"
echo -e "  Data Dir:     ${YELLOW}$DATA_DIR${NC}"
echo ""

# Function to evaluate ES model
evaluate_es() {
    echo -e "${BLUE}────────────────────────────────────────────${NC}"
    echo -e "${BLUE}Evaluating ES Model${NC}"
    echo -e "${BLUE}────────────────────────────────────────────${NC}"
    
    # Auto-detect checkpoint directory if not provided
    if [ -z "$CHECKPOINT_DIR" ]; then
        if [ "$TASK" == "gsm8k" ]; then
            ES_CHECKPOINT="./checkpoints/es_gsm8k_${MODEL_TYPE}_split${TRAIN_SPLIT}"
        else
            ES_CHECKPOINT="./checkpoints/es_countdown_split${TRAIN_SPLIT}"
        fi
    else
        ES_CHECKPOINT="$CHECKPOINT_DIR"
    fi
    
    echo -e "Checkpoint:   ${YELLOW}$ES_CHECKPOINT${NC}"
    
    # Check if checkpoint exists
    if [ ! -d "$ES_CHECKPOINT" ]; then
        echo -e "${RED}Error: ES checkpoint directory not found: $ES_CHECKPOINT${NC}"
        return 1
    fi
    
    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(ls -d "$ES_CHECKPOINT"/iteration_* 2>/dev/null | sort -V | tail -n 1)
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo -e "${YELLOW}Warning: No iteration checkpoints found, using base checkpoint${NC}"
        LATEST_CHECKPOINT="$ES_CHECKPOINT"
    else
        echo -e "Latest checkpoint: ${YELLOW}$(basename $LATEST_CHECKPOINT)${NC}"
    fi
    
    # Determine output file
    MODEL_BASENAME=$(echo "$MODEL_NAME" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
    OUTPUT_FILE="./src/evals/es-evals/${MODEL_BASENAME}_${MODEL_TYPE}_eval_results_${TASK}_es_${TRAIN_SPLIT}.json"
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    
    echo -e "Output:       ${YELLOW}$OUTPUT_FILE${NC}"
    echo ""
    
    # Run evaluation using vLLM
    if [ "$TASK" == "gsm8k" ]; then
        python ./src/scripts/evaluation/eval_gsm8k_vllm.py \
            --model_path "$LATEST_CHECKPOINT" \
            --data_file "$DATA_DIR/${EVAL_SPLIT}.parquet" \
            --output_file "$OUTPUT_FILE" \
            --batch_size "$BATCH_SIZE" \
            --num_gpus "$NUM_GPUS"
    else
        python ./src/scripts/evaluation/eval_countdown_vllm.py \
            --model_path "$LATEST_CHECKPOINT" \
            --data_file "$DATA_DIR/${EVAL_SPLIT}.parquet" \
            --output_file "$OUTPUT_FILE" \
            --batch_size "$BATCH_SIZE" \
            --num_gpus "$NUM_GPUS"
    fi
    
    echo -e "${GREEN}✓ ES Evaluation complete${NC}"
    echo -e "  Results saved to: ${YELLOW}$OUTPUT_FILE${NC}"
    echo ""
}

# Function to evaluate GRPO model
evaluate_grpo() {
    echo -e "${BLUE}────────────────────────────────────────────${NC}"
    echo -e "${BLUE}Evaluating GRPO Model${NC}"
    echo -e "${BLUE}────────────────────────────────────────────${NC}"
    
    # Auto-detect checkpoint directory if not provided
    if [ -z "$CHECKPOINT_DIR" ]; then
        if [ "$TASK" == "gsm8k" ]; then
            GRPO_CHECKPOINT="./checkpoints/verl_grpo_gsm8k_${MODEL_TYPE}"
        else
            GRPO_CHECKPOINT="./checkpoints/verl_grpo_countdown"
        fi
    else
        GRPO_CHECKPOINT="$CHECKPOINT_DIR"
    fi
    
    echo -e "Checkpoint:   ${YELLOW}$GRPO_CHECKPOINT${NC}"
    
    # Check if checkpoint exists
    if [ ! -d "$GRPO_CHECKPOINT" ]; then
        echo -e "${RED}Error: GRPO checkpoint directory not found: $GRPO_CHECKPOINT${NC}"
        return 1
    fi
    
    # Determine output file
    MODEL_BASENAME=$(echo "$MODEL_NAME" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
    OUTPUT_FILE="./src/evals/${MODEL_BASENAME}_${MODEL_TYPE}_eval_results_${TASK}_${TRAIN_SPLIT}.json"
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    
    echo -e "Output:       ${YELLOW}$OUTPUT_FILE${NC}"
    echo ""
    
    # Run evaluation script (this will merge checkpoints and evaluate)
    if [ "$TASK" == "gsm8k" ]; then
        bash ./src/scripts/evaluation/evaluate_gsm8k.sh \
            --checkpoint_dir "$GRPO_CHECKPOINT" \
            --data_file "$DATA_DIR/${EVAL_SPLIT}.parquet" \
            --output_file "$OUTPUT_FILE" \
            --batch_size "$BATCH_SIZE"
    else
        bash ./src/scripts/evaluation/evaluate_countdown.sh \
            --checkpoint_dir "$GRPO_CHECKPOINT" \
            --data_file "$DATA_DIR/${EVAL_SPLIT}.parquet" \
            --output_file "$OUTPUT_FILE" \
            --batch_size "$BATCH_SIZE"
    fi
    
    echo -e "${GREEN}✓ GRPO Evaluation complete${NC}"
    echo -e "  Results saved to: ${YELLOW}$OUTPUT_FILE${NC}"
    echo ""
}

# Run evaluations
if [ "$METHOD" == "es" ]; then
    evaluate_es
elif [ "$METHOD" == "grpo" ]; then
    evaluate_grpo
elif [ "$METHOD" == "both" ]; then
    evaluate_es
    evaluate_grpo
    
    # Generate comparison summary
    echo -e "${BLUE}────────────────────────────────────────────${NC}"
    echo -e "${BLUE}Generating Comparison Summary${NC}"
    echo -e "${BLUE}────────────────────────────────────────────${NC}"
    
    # This would call a Python script to compare results
    # python ./src/scripts/evaluation/compare_results.py \
    #     --es_results "$OUTPUT_FILE_ES" \
    #     --grpo_results "$OUTPUT_FILE_GRPO" \
    #     --output "./src/evals/comparison_${TASK}_${TRAIN_SPLIT}.json"
    
    echo -e "${YELLOW}Note: Run generate_charts.py to visualize comparison${NC}"
    echo ""
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Evaluation Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "Method:       ${YELLOW}$METHOD${NC}"
echo -e "Task:         ${YELLOW}$TASK${NC}"
echo -e "Train Split:  ${YELLOW}$TRAIN_SPLIT${NC}"
echo ""
echo -e "${YELLOW}Results location:${NC}"
echo -e "  ${YELLOW}./src/evals/${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Review results in evaluation output files"
echo -e "2. Generate visualizations:"
echo -e "   ${YELLOW}python src/scripts/generate_charts.py${NC}"
echo -e "3. Upload results to HuggingFace:"
echo -e "   ${YELLOW}python upload_inference_results.py${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
