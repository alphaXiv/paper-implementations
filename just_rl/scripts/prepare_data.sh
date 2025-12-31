#!/bin/bash
# just_rl/scripts/prepare_data.sh

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Use Python from the virtual environment
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at ${PROJECT_ROOT}/.venv"
    echo "Please create a virtual environment and install dependencies first."
    exit 1
fi

# Add project root to PYTHONPATH to ensure imports work
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

# Define paths
DATA_OUTPUT_DIR="${PROJECT_ROOT}/data/processed"
DATASET_NAME="BytedTsinghua-SIA/DAPO-Math-17k"  # As per paper/requirements
DAPO_LOADER="${PROJECT_ROOT}/data/dapo_loader.py"

echo "=================================================================="
echo "JustRL Data Preparation"
echo "=================================================================="
echo "Project Root: ${PROJECT_ROOT}"
echo "Output Directory: ${DATA_OUTPUT_DIR}"
echo "Dataset: ${DATASET_NAME}"
echo "=================================================================="

# Create output directory if it doesn't exist
mkdir -p "$DATA_OUTPUT_DIR"

# Run the python loader directly as a script using venv Python
echo "Running dapo_loader.py..."
"$VENV_PYTHON" "$DAPO_LOADER" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$DATA_OUTPUT_DIR"

echo "=================================================================="
echo "Data preparation complete!"
echo "Check ${DATA_OUTPUT_DIR} for the processed parquet file."
echo "=================================================================="
