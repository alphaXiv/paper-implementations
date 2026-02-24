#!/bin/bash
# Speedrun script for GRPO training on GSM8K and Countdown tasks
# Handles data preparation and training execution with proper environment setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
TASK=""
MODEL_TYPE="instruct"  # or "base" for GSM8K
TRAIN_SPLIT=0.1
TEST_SAMPLES=200
SKIP_DATA_PREP=false
HF_TOKEN_VALUE=""

# Print usage
usage() {
    cat << EOF
Usage: $0 --task <gsm8k|countdown> [OPTIONS]

Required:
  --task TASK              Task to run: 'gsm8k' or 'countdown'

Optional:
  --model MODEL_TYPE       Model type for GSM8K: 'instruct' (default) or 'base'
  --train-split FRACTION   Fraction of data for training (default: 0.1)
  --test-samples N         Number of samples reserved for test (default: 200)
  --skip-data-prep         Skip data preparation step
  --hf-token TOKEN         HuggingFace token (or set HF_TOKEN env var)
  --help                   Show this help message

Examples:
  # GSM8K with instruct model
  export HF_TOKEN=your_token_here
  $0 --task gsm8k --train-split 0.4

  # GSM8K with base model
  $0 --task gsm8k --model base --hf-token hf_xxx...

  # Countdown task
  $0 --task countdown --train-split 0.1 --test-samples 200

  # Docker execution
  sudo docker exec -e HF_TOKEN=hf_xxx -it verl-es-fine-tuning-paper bash -c \\
    "cd es-fine-tuning-paper/src && bash speedrun.sh --task gsm8k"

EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --train-split)
            TRAIN_SPLIT="$2"
            shift 2
            ;;
        --test-samples)
            TEST_SAMPLES="$2"
            shift 2
            ;;
        --skip-data-prep)
            SKIP_DATA_PREP=true
            shift
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

# Validate task
if [ -z "$TASK" ]; then
    echo -e "${RED}Error: --task is required${NC}"
    usage
fi

if [ "$TASK" != "gsm8k" ] && [ "$TASK" != "countdown" ]; then
    echo -e "${RED}Error: --task must be 'gsm8k' or 'countdown'${NC}"
    usage
fi

# Validate model type
if [ "$TASK" == "gsm8k" ] && [ "$MODEL_TYPE" != "instruct" ] && [ "$MODEL_TYPE" != "base" ]; then
    echo -e "${RED}Error: --model must be 'instruct' or 'base' for GSM8K${NC}"
    usage
fi

# Set HF_TOKEN if provided via flag
if [ -n "$HF_TOKEN_VALUE" ]; then
    export HF_TOKEN="$HF_TOKEN_VALUE"
fi

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}Error: HF_TOKEN environment variable is not set${NC}"
    echo -e "${YELLOW}Please run: export HF_TOKEN=your_huggingface_token${NC}"
    echo -e "${YELLOW}Or use: $0 --task $TASK --hf-token your_token${NC}"
    exit 1
fi

# Print configuration
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}GRPO Training Speedrun${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "Task: ${YELLOW}$TASK${NC}"
if [ "$TASK" == "gsm8k" ]; then
    echo -e "Model type: ${YELLOW}$MODEL_TYPE${NC}"
fi
echo -e "Train split: ${YELLOW}$TRAIN_SPLIT${NC}"
echo -e "Test samples (reserved): ${YELLOW}$TEST_SAMPLES${NC}"
echo -e "Skip data prep: ${YELLOW}$SKIP_DATA_PREP${NC}"
echo -e "HF Token: ${YELLOW}${HF_TOKEN:0:10}...${NC}"
echo ""

# Step 1: Data Preparation
if [ "$SKIP_DATA_PREP" == "false" ]; then
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}Step 1: Data Preparation${NC}"
    echo -e "${GREEN}=========================================${NC}"
    
    if [ "$TASK" == "gsm8k" ]; then
        DATA_DIR="./data/gsm8k-$TRAIN_SPLIT"
        echo -e "Preparing GSM8K data..."
        echo -e "Output directory: ${YELLOW}$DATA_DIR${NC}"
        
        ./prepare_gsm8k_data.sh \
            --local_dir "$DATA_DIR" \
            --train_split "$TRAIN_SPLIT" \
            --test_samples "$TEST_SAMPLES"
        
    elif [ "$TASK" == "countdown" ]; then
        DATA_DIR="./data/countdown-$TRAIN_SPLIT"
        echo -e "Preparing Countdown data..."
        echo -e "Output directory: ${YELLOW}$DATA_DIR${NC}"
        
        ./prepare_countdown_data.sh \
            --local_dir "$DATA_DIR" \
            --train_split "$TRAIN_SPLIT" \
            --test_samples "$TEST_SAMPLES"
    fi
    
    echo -e "${GREEN}✓ Data preparation complete${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping data preparation${NC}"
    if [ "$TASK" == "gsm8k" ]; then
        DATA_DIR="./data/gsm8k-$TRAIN_SPLIT"
    elif [ "$TASK" == "countdown" ]; then
        DATA_DIR="./data/countdown-$TRAIN_SPLIT"
    fi
    echo -e "Expected data directory: ${YELLOW}$DATA_DIR${NC}"
    echo ""
fi

# Step 2: Training
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Step 2: GRPO Training${NC}"
echo -e "${GREEN}=========================================${NC}"

if [ "$TASK" == "gsm8k" ]; then
    if [ "$MODEL_TYPE" == "base" ]; then
        echo -e "Running GSM8K training with ${YELLOW}base model${NC}..."
        echo -e "Training script: ${YELLOW}grpo-gsm8k-base.sh${NC}"
        
        # Update grpo-gsm8k-base.sh to use the correct data directory
        sed -i "s|data.train_files=./data/gsm8k-[^/]*/train.parquet|data.train_files=$DATA_DIR/train.parquet|g" grpo-gsm8k-base.sh
        sed -i "s|data.val_files=./data/gsm8k-[^/]*/validation.parquet|data.val_files=$DATA_DIR/validation.parquet|g" grpo-gsm8k-base.sh
        
        bash grpo-gsm8k-base.sh
        
    else  # instruct
        echo -e "Running GSM8K training with ${YELLOW}instruct model${NC}..."
        echo -e "Training script: ${YELLOW}grpo-gsm8k.sh${NC}"
        
        # Update grpo-gsm8k.sh to use the correct data directory
        sed -i "s|data.train_files=./data/gsm8k-[^/]*/train.parquet|data.train_files=$DATA_DIR/train.parquet|g" grpo-gsm8k.sh
        sed -i "s|data.val_files=./data/gsm8k-[^/]*/validation.parquet|data.val_files=$DATA_DIR/validation.parquet|g" grpo-gsm8k.sh
        
        bash grpo-gsm8k.sh
    fi
    
elif [ "$TASK" == "countdown" ]; then
    echo -e "Running Countdown training..."
    echo -e "Training script: ${YELLOW}grpo-countdown-custom.sh${NC}"
    
    # Update grpo-countdown-custom.sh to use the correct data directory
    sed -i "s|data.train_files=./data/countdown-[^/]*/train.parquet|data.train_files=$DATA_DIR/train.parquet|g" grpo-countdown-custom.sh
    sed -i "s|data.val_files=./data/countdown-[^/]*/|data.val_files=$DATA_DIR/validation.parquet|g" grpo-countdown-custom.sh
    
    bash grpo-countdown-custom.sh
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}✓ Training Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "Task: ${YELLOW}$TASK${NC}"
if [ "$TASK" == "gsm8k" ]; then
    echo -e "Model: ${YELLOW}$MODEL_TYPE${NC}"
    CHECKPOINT_DIR="./checkpoints/verl_grpo_gsm8k_${MODEL_TYPE}/llama3.2_3b_${MODEL_TYPE}_grpo_lora"
else
    CHECKPOINT_DIR="./checkpoints/verl_grpo_countdown/llama3.2_3b_countdown_grpo_lora"
fi
echo -e "Data directory: ${YELLOW}$DATA_DIR${NC}"
echo -e "Checkpoints: ${YELLOW}$CHECKPOINT_DIR${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Merge FSDP checkpoints:"
if [ "$TASK" == "gsm8k" ]; then
    echo -e "   ${YELLOW}bash evaluate_gsm8k.sh${NC}"
else
    echo -e "   ${YELLOW}bash evaluate_countdown.sh${NC}"
fi
echo -e "2. Evaluate on reserved test set"
echo -e "${GREEN}=========================================${NC}"
