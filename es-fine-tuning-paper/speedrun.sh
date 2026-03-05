#!/bin/bash
# Speedrun script for ES and GRPO training on GSM8K and Countdown tasks
# Uses Docker environment setup via verl-docker-run.sh
# Handles data preparation and training execution with proper environment setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
METHOD=""           # "es" or "grpo"
TASK=""            # "gsm8k" or "countdown"
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
MODEL_TYPE="instruct"
TRAIN_SPLIT=0.1
NUM_SAMPLES=700    # For ES training
TEST_SAMPLES=200
POPULATION_SIZE=8  # For ES
NUM_ITERATIONS=100 # For ES
CUDA_DEVICES="0,1,2,3"
NUM_ENGINES=4      # For ES vLLM engines
SKIP_DATA_PREP=false
SKIP_DOCKER_SETUP=false
HF_TOKEN_VALUE=""
WANDB_API_KEY_VALUE=""

# Print usage
usage() {
    cat << EOF
${GREEN}ES vs GRPO Speedrun Script${NC}
Usage: $0 --method <es|grpo> --task <gsm8k|countdown> [OPTIONS]

${YELLOW}Required:${NC}
  --method METHOD          Training method: 'es' or 'grpo' or 'both'
  --task TASK              Task to run: 'gsm8k' or 'countdown'

${YELLOW}Optional - General:${NC}
  --model MODEL_NAME       Model name (default: Qwen/Qwen2.5-3B-Instruct)
  --model-type TYPE        Model type: 'instruct' or 'base' (default: instruct)
  --train-split FRACTION   Fraction of data for training (default: 0.1)
  --test-samples N         Number of samples reserved for test (default: 200)
  --skip-data-prep         Skip data preparation step
  --skip-docker            Skip Docker environment setup
  --hf-token TOKEN         HuggingFace token (or set HF_TOKEN env var)
  --wandb-key KEY          Weights & Biases API key (or set WANDB_API_KEY env var)

${YELLOW}Optional - ES Specific:${NC}
  --num-samples N          Number of training samples for ES (default: 700)
  --population-size N      ES population size (default: 8)
  --num-iterations N       ES iterations (default: 100)
  --cuda-devices DEVICES   CUDA devices for ES (default: 0,1,2,3)
  --num-engines N          Number of vLLM engines for ES (default: 4)

${YELLOW}Examples:${NC}
  # Run ES training on GSM8K with 10% data
  $0 --method es --task gsm8k --train-split 0.1 --num-samples 700

  # Run GRPO training on GSM8K with 40% data
  $0 --method grpo --task gsm8k --train-split 0.4

  # Run both ES and GRPO on Countdown
  $0 --method both --task countdown --train-split 0.4

  # Run with base model
  $0 --method grpo --task gsm8k --model-type base --model Qwen/Qwen2.5-3B

  # Skip Docker setup (if already running)
  $0 --method es --task gsm8k --skip-docker

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
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --train-split)
            TRAIN_SPLIT="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --test-samples)
            TEST_SAMPLES="$2"
            shift 2
            ;;
        --population-size)
            POPULATION_SIZE="$2"
            shift 2
            ;;
        --num-iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --num-engines)
            NUM_ENGINES="$2"
            shift 2
            ;;
        --skip-data-prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        --skip-docker)
            SKIP_DOCKER_SETUP=true
            shift
            ;;
        --hf-token)
            HF_TOKEN_VALUE="$2"
            shift 2
            ;;
        --wandb-key)
            WANDB_API_KEY_VALUE="$2"
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

# Validate model type
if [ "$MODEL_TYPE" != "instruct" ] && [ "$MODEL_TYPE" != "base" ]; then
    echo -e "${RED}Error: --model-type must be 'instruct' or 'base'${NC}"
    usage
fi

# Set environment variables
if [ -n "$HF_TOKEN_VALUE" ]; then
    export HF_TOKEN="$HF_TOKEN_VALUE"
fi

if [ -n "$WANDB_API_KEY_VALUE" ]; then
    export WANDB_API_KEY="$WANDB_API_KEY_VALUE"
fi

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}Error: HF_TOKEN not set. Use --hf-token or set HF_TOKEN environment variable${NC}"
    exit 1
fi

if [ "$METHOD" == "grpo" ] || [ "$METHOD" == "both" ]; then
    if [ -z "$WANDB_API_KEY" ]; then
        echo -e "${YELLOW}Warning: WANDB_API_KEY not set. GRPO training may fail without it.${NC}"
        echo -e "${YELLOW}Use --wandb-key or set WANDB_API_KEY environment variable${NC}"
    fi
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        ES vs GRPO Training Speedrun                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Method:       ${YELLOW}$METHOD${NC}"
echo -e "  Task:         ${YELLOW}$TASK${NC}"
echo -e "  Model:        ${YELLOW}$MODEL_NAME${NC}"
echo -e "  Model Type:   ${YELLOW}$MODEL_TYPE${NC}"
echo -e "  Train Split:  ${YELLOW}$TRAIN_SPLIT${NC}"
echo -e "  Test Samples: ${YELLOW}$TEST_SAMPLES${NC}"
if [ "$METHOD" == "es" ] || [ "$METHOD" == "both" ]; then
    echo -e "  ES Samples:   ${YELLOW}$NUM_SAMPLES${NC}"
    echo -e "  Population:   ${YELLOW}$POPULATION_SIZE${NC}"
    echo -e "  Iterations:   ${YELLOW}$NUM_ITERATIONS${NC}"
    echo -e "  CUDA Devices: ${YELLOW}$CUDA_DEVICES${NC}"
fi
echo ""

# Step 0: Docker environment setup (for GRPO)
if [ "$METHOD" == "grpo" ] || [ "$METHOD" == "both" ]; then
    if [ "$SKIP_DOCKER_SETUP" = false ]; then
        echo -e "${GREEN}[Step 0] Setting up Docker environment for GRPO...${NC}"
        if [ ! -f "./verl-docker-run.sh" ]; then
            echo -e "${RED}Error: verl-docker-run.sh not found${NC}"
            exit 1
        fi
        bash ./verl-docker-run.sh
        echo -e "${GREEN}✓ Docker environment ready${NC}"
        echo ""
    else
        echo -e "${GREEN}[Step 0] Skipping Docker setup (using existing container)${NC}"
        # Verify container is running
        if ! sudo docker ps | grep -q "verl-es-fine-tuning-paper"; then
            echo -e "${RED}Error: Docker container 'verl-es-fine-tuning-paper' not running${NC}"
            echo -e "${YELLOW}Run without --skip-docker to set it up${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Container verified${NC}"
        echo ""
    fi
else
    echo -e "${GREEN}[Step 0] Skipping Docker setup (ES-only mode)${NC}"
    echo ""
fi

# Step 1: Data Preparation
if [ "$SKIP_DATA_PREP" == "false" ]; then
    echo -e "${GREEN}[Step 1] Data Preparation${NC}"
    
    if [ "$TASK" == "gsm8k" ]; then
        DATA_DIR="./src/data/gsm8k-$TRAIN_SPLIT"
        echo -e "Preparing GSM8K data..."
        echo -e "Output directory: ${YELLOW}$DATA_DIR${NC}"
        
        bash ./src/scripts/data_prep/prepare_gsm8k_data.sh \
            --local_dir "$DATA_DIR" \
            --train_split "$TRAIN_SPLIT" \
            --test_samples "$TEST_SAMPLES"
        
    elif [ "$TASK" == "countdown" ]; then
        DATA_DIR="./src/data/countdown-$TRAIN_SPLIT"
        echo -e "Preparing Countdown data..."
        echo -e "Output directory: ${YELLOW}$DATA_DIR${NC}"
        
        bash ./src/scripts/data_prep/prepare_countdown_data.sh \
            --local_dir "$DATA_DIR" \
            --train_split "$TRAIN_SPLIT" \
            --test_samples "$TEST_SAMPLES"
    fi
    
    echo -e "${GREEN}✓ Data preparation complete${NC}"
    echo ""
else
    echo -e "${YELLOW}[Step 1] Skipping data preparation${NC}"
    if [ "$TASK" == "gsm8k" ]; then
        DATA_DIR="./src/data/gsm8k-$TRAIN_SPLIT"
    elif [ "$TASK" == "countdown" ]; then
        DATA_DIR="./src/data/countdown-$TRAIN_SPLIT"
    fi
    echo -e "Expected data directory: ${YELLOW}$DATA_DIR${NC}"
    echo ""
fi

# Step 2: Training
echo -e "${GREEN}[Step 2] Training${NC}"

# ES Training
if [ "$METHOD" == "es" ] || [ "$METHOD" == "both" ]; then
    echo -e "${BLUE}Running ES Training...${NC}"
    
    if [ "$TASK" == "gsm8k" ]; then
        ES_SCRIPT="./src/scripts/es/es_fine_tuning_gsm8k_accl.py"
        echo -e "ES Script: ${YELLOW}$ES_SCRIPT${NC}"
        
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python "$ES_SCRIPT" \
            --model_name "$MODEL_NAME" \
            --data_dir "$DATA_DIR" \
            --num_samples "$NUM_SAMPLES" \
            --population_size "$POPULATION_SIZE" \
            --num_iterations "$NUM_ITERATIONS" \
            --num_engines "$NUM_ENGINES" \
            --output_dir "./checkpoints/es_gsm8k_${MODEL_TYPE}_split${TRAIN_SPLIT}"
        
    elif [ "$TASK" == "countdown" ]; then
        ES_SCRIPT="./src/scripts/es/es_fine_tuning_countdown_accl.py"
        echo -e "ES Script: ${YELLOW}$ES_SCRIPT${NC}"
        
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python "$ES_SCRIPT" \
            --model_name "$MODEL_NAME" \
            --data_dir "$DATA_DIR" \
            --num_samples "$NUM_SAMPLES" \
            --population_size "$POPULATION_SIZE" \
            --num_iterations "$NUM_ITERATIONS" \
            --num_engines "$NUM_ENGINES" \
            --output_dir "./checkpoints/es_countdown_split${TRAIN_SPLIT}"
    fi
    
    echo -e "${GREEN}✓ ES Training complete${NC}"
    echo ""
fi

# GRPO Training  
if [ "$METHOD" == "grpo" ] || [ "$METHOD" == "both" ]; then
    echo -e "${BLUE}Running GRPO Training (in Docker)...${NC}"
    
    if [ "$TASK" == "gsm8k" ]; then
        if [ "$MODEL_TYPE" == "base" ]; then
            GRPO_SCRIPT="./src/scripts/grpo/grpo-gsm8k-base.sh"
            echo -e "GRPO Script: ${YELLOW}$GRPO_SCRIPT${NC}"
            
            # Update data paths in script
            sudo docker exec verl-es-fine-tuning-paper bash -c \
                "cd /workspace/es-fine-tuning-paper && \
                 sed -i \"s|data.train_files=.*train.parquet|data.train_files=$DATA_DIR/train.parquet|g\" $GRPO_SCRIPT && \
                 sed -i \"s|data.val_files=.*validation.parquet|data.val_files=$DATA_DIR/validation.parquet|g\" $GRPO_SCRIPT"
            
        else  # instruct
            GRPO_SCRIPT="./src/scripts/grpo/grpo-gsm8k.sh"
            echo -e "GRPO Script: ${YELLOW}$GRPO_SCRIPT${NC}"
            
            # Update data paths in script
            sudo docker exec verl-es-fine-tuning-paper bash -c \
                "cd /workspace/es-fine-tuning-paper && \
                 sed -i \"s|data.train_files=.*train.parquet|data.train_files=$DATA_DIR/train.parquet|g\" $GRPO_SCRIPT && \
                 sed -i \"s|data.val_files=.*validation.parquet|data.val_files=$DATA_DIR/validation.parquet|g\" $GRPO_SCRIPT"
        fi
        
        # Run GRPO training with environment variables
        sudo docker exec \
            -e HF_TOKEN="$HF_TOKEN" \
            -e WANDB_API_KEY="$WANDB_API_KEY" \
            verl-es-fine-tuning-paper bash -c \
            "cd /workspace/es-fine-tuning-paper && bash $GRPO_SCRIPT"
        
    elif [ "$TASK" == "countdown" ]; then
        GRPO_SCRIPT="./src/scripts/grpo/grpo-countdown-custom.sh"
        echo -e "GRPO Script: ${YELLOW}$GRPO_SCRIPT${NC}"
        
        # Update data paths in script
        sudo docker exec verl-es-fine-tuning-paper bash -c \
            "cd /workspace/es-fine-tuning-paper && \
             sed -i \"s|data.train_files=.*train.parquet|data.train_files=$DATA_DIR/train.parquet|g\" $GRPO_SCRIPT && \
             sed -i \"s|data.val_files=.*validation.parquet|data.val_files=$DATA_DIR/validation.parquet|g\" $GRPO_SCRIPT"
        
        # Run GRPO training with environment variables
        sudo docker exec \
            -e HF_TOKEN="$HF_TOKEN" \
            -e WANDB_API_KEY="$WANDB_API_KEY" \
            verl-es-fine-tuning-paper bash -c \
            "cd /workspace/es-fine-tuning-paper && bash $GRPO_SCRIPT"
    fi
    
    echo -e "${GREEN}✓ GRPO Training complete${NC}"
    echo ""
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}✓ All Training Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "Method:       ${YELLOW}$METHOD${NC}"
echo -e "Task:         ${YELLOW}$TASK${NC}"
echo -e "Model Type:   ${YELLOW}$MODEL_TYPE${NC}"
echo -e "Data Dir:     ${YELLOW}$DATA_DIR${NC}"
echo ""

# Show checkpoint locations
if [ "$METHOD" == "es" ] || [ "$METHOD" == "both" ]; then
    if [ "$TASK" == "gsm8k" ]; then
        ES_CHECKPOINT="./checkpoints/es_gsm8k_${MODEL_TYPE}_split${TRAIN_SPLIT}"
    else
        ES_CHECKPOINT="./checkpoints/es_countdown_split${TRAIN_SPLIT}"
    fi
    echo -e "ES Checkpoints:   ${YELLOW}$ES_CHECKPOINT${NC}"
fi

if [ "$METHOD" == "grpo" ] || [ "$METHOD" == "both" ]; then
    if [ "$TASK" == "gsm8k" ]; then
        GRPO_CHECKPOINT="./checkpoints/verl_grpo_gsm8k_${MODEL_TYPE}"
    else
        GRPO_CHECKPOINT="./checkpoints/verl_grpo_countdown"
    fi
    echo -e "GRPO Checkpoints: ${YELLOW}$GRPO_CHECKPOINT${NC}"
fi

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Run evaluations:"
echo -e "   ${YELLOW}bash evaluation.sh --method $METHOD --task $TASK --train-split $TRAIN_SPLIT${NC}"
echo -e "2. View results in ${YELLOW}./src/evals/${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
