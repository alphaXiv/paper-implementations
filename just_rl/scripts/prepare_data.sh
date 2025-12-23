#!/bin/bash
# justrl_reproduction/scripts/prepare_data.sh

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add project root to PYTHONPATH to ensure imports work
export PYTHONPATH="${PROJECT_ROOT}/..:$PYTHONPATH"

# Define paths
DATA_OUTPUT_DIR="${PROJECT_ROOT}/data/processed"
DATASET_NAME="dapo-ai/DAPO-Math-17k"  # As per paper/requirements

echo "=================================================================="
echo "JustRL Data Preparation"
echo "=================================================================="
echo "Project Root: ${PROJECT_ROOT}"
echo "Output Directory: ${DATA_OUTPUT_DIR}"
echo "Dataset: ${DATASET_NAME}"
echo "=================================================================="

# Create output directory if it doesn't exist
mkdir -p "$DATA_OUTPUT_DIR"

# Run the python loader
# We use the module syntax to ensure relative imports within the package work correctly
echo "Running dapo_loader.py..."
python3 -m justrl_reproduction.data.dapo_loader \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$DATA_OUTPUT_DIR"

echo "=================================================================="
echo "Data preparation complete!"
echo "Check ${DATA_OUTPUT_DIR} for the processed parquet file."
echo "=================================================================="
