#!/bin/bash
# Script to prepare Countdown data for GRPO training with reserved test set

set -e

echo "=========================================="
echo "Preparing Countdown data for GRPO training"
echo "=========================================="

# Default parameters
LOCAL_DIR="./data/countdown-0.4"
JSON_FILE="../countdown/data/countdown.json"
TRAIN_SPLIT=0.4
TEST_SAMPLES=200

# Install dependencies
pip install pandas

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local_dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        --json_file)
            JSON_FILE="$2"
            shift 2
            ;;
        --train_split)
            TRAIN_SPLIT="$2"
            shift 2
            ;;
        --test_samples)
            TEST_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--local_dir DIR] [--json_file FILE] [--train_split FRACTION] [--test_samples N]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Input JSON file: $JSON_FILE"
echo "  Output directory: $LOCAL_DIR"
echo "  Train split: $TRAIN_SPLIT (${TRAIN_SPLIT%.*}0% of available data)"
echo "  Test samples (reserved): $TEST_SAMPLES"
echo ""

# Run the data preparation script
python3 grpo_data_countdown.py \
    --local_dir "$LOCAL_DIR" \
    --json_file "$JSON_FILE" \
    --train_split "$TRAIN_SPLIT" \
    --test_samples "$TEST_SAMPLES"

echo ""
echo "Data preparation complete!"
echo "You can now run GRPO training with: ./grpo-countdown.sh"
