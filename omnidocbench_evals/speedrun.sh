#!/bin/bash

# OmniDocBench-Evals Speedrun Script
# Sets up and runs evaluation for DeepSeek-OCR, OLMOCR-2, and Chandra OCR on OmniDocBench
# Usage: ./speedrun.sh [language_filter]
# Options: all, english, simplified_chinese (default: all)

set -e

# Parse language filter argument
LANGUAGE_FILTER="${1:-all}"

if [[ ! "$LANGUAGE_FILTER" =~ ^(all|english|simplified_chinese)$ ]]; then
    echo "Invalid language filter: $LANGUAGE_FILTER"
    echo "Usage: ./speedrun.sh [all|english|simplified_chinese]"
    echo "Default: all"
    exit 1
fi

echo "=========================================="
echo "OmniDocBench-Evals Setup and Evaluation"
echo "Language Filter: $LANGUAGE_FILTER"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda or Anaconda from https://docs.conda.io/projects/miniconda/en/latest/"
    echo "After installation, run: conda init bash && source ~/.bashrc"
    exit 1
else
    echo "Conda is already installed."
fi

# Ensure conda is available
export PATH="$HOME/miniconda/bin:$PATH"

# Create data directory
mkdir -p data

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads."
fi

# # Download OmniDocBench dataset
# echo "Downloading OmniDocBench dataset..."
# conda run -n base pip install huggingface_hub
# python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='opendatalab/OmniDocBench', repo_type='dataset', local_dir='data/OmniDocBench')"
# echo "Dataset downloaded to data/OmniDocBench"

# # Convert images to PDFs for OLMOCR
# echo "Converting images to PDFs for OLMOCR..."
# Install basic dependencies
conda run -n base pip install PyMuPDF tqdm 

# Get absolute path
ABS_PATH=$(pwd)

# # Assume the images are in data/OmniDocBench/images/
# # Edit the script to set input_dir and output_dir

# sed -i "s|input_directory = .*|input_directory = \"$ABS_PATH/data/OmniDocBench/images/\"|" src/omnidocbench_evals/OmniDocBench/tools/image_to_pdf.py
# sed -i "s|output_directory = .*|output_directory = \"$ABS_PATH/data/OmniDocBench/pdfs/\"|" src/omnidocbench_evals/OmniDocBench/tools/image_to_pdf.py
# python src/omnidocbench_evals/OmniDocBench/tools/image_to_pdf.py
# echo "PDFs created."

# # Setup DeepSeek-OCR
# echo "Setting up DeepSeek-OCR..."

# conda create -n deepseek-ocr python=3.12.9 -y
# conda run -n deepseek-ocr pip install -e .[deepseek-ocr] --extra-index-url https://download.pytorch.org/whl/cu118
# conda run -n deepseek-ocr pip install flash_attn==2.7.3 --no-build-isolation

# cd src/omnidocbench_evals/DeepSeek-OCR-master/DeepSeek-OCR-vllm
# # Configure config.py with absolute paths
# DEEPSEEK_ABS_PATH="$ABS_PATH/src/omnidocbench_evals/DeepSeek-OCR-master/DeepSeek-OCR-vllm"
# sed -i "s|INPUT_PATH = 'data/OmniDocBench/images/'|INPUT_PATH = '$ABS_PATH/data/OmniDocBench/images/'|" config.py
# sed -i "s|OUTPUT_PATH = 'outputs/deepseek_ocr/'|OUTPUT_PATH = '$ABS_PATH/outputs/deepseek_ocr/'|" config.py
# echo "DeepSeek-OCR setup complete."

# # Run DeepSeek-OCR inference
# echo "Running DeepSeek-OCR inference..."
# conda run -n deepseek-ocr python run_dpsk_ocr_eval_batch.py
# echo "DeepSeek-OCR inference complete. Outputs in outputs/deepseek_ocr/"

# echo "Cleaning up the outputs/deepseek_ocr/ directory..."
# rm -f $ABS_PATH/outputs/deepseek_ocr/*.png

# echo "Cleaning complete. Now setting up olmOCR2.."

cd $ABS_PATH/src/omnidocbench_evals/olmocr

# echo "=========================================="

# Setup OLMOCR
echo "Setting up OLMOCR..."
sudo apt-get update
sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
conda create -n olmocr python=3.11 -y
# Install olmocr from source
conda run -n olmocr pip install -e $ABS_PATH/src/omnidocbench_evals/olmocr
conda run -n olmocr pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128

echo "OLMOCR setup complete."

# Run OLMOCR inference
echo "Running OLMOCR inference..."
# Assume PDFs are in data/OmniDocBench/pdfs/ (created by image_to_pdf.py)
mkdir -p $ABS_PATH/outputs/olmocr_workspace
# Create a text file listing all PDFs
find $ABS_PATH/data/OmniDocBench/pdfs/ -name "*.pdf" -type f > $ABS_PATH/outputs/olmocr_workspace/pdf_list.txt
echo "Found $(wc -l < $ABS_PATH/outputs/olmocr_workspace/pdf_list.txt) PDF files to process"
# Use eval to properly activate the conda environment so vllm is in PATH
eval "$(conda shell.bash hook)"
conda activate olmocr
python -m olmocr.pipeline $ABS_PATH/outputs/olmocr_workspace --markdown --pdfs $ABS_PATH/outputs/olmocr_workspace/pdf_list.txt
conda deactivate
echo "OLMOCR inference complete. Converting JSONL to markdown..."
# Convert JSONL results to markdown files
python $ABS_PATH/src/omnidocbench_evals/olmocr/convert_jsonl_to_markdown.py
echo "Markdown files created in outputs/olmocr_workspace/markdown/"

cd $ABS_PATH

echo "=========================================="

# Setup Chandra OCR
echo "Setting up Chandra OCR..."
# Reuse olmocr env to avoid conflicts
conda run -n olmocr pip install -e .[chandra-ocr]
cd $ABS_PATH/src/omnidocbench_evals/chandra-ocr
echo "Chandra OCR setup complete."

# Check for API key
if [ -z "$DATALAB_API_KEY" ]; then
    echo "WARNING: DATALAB_API_KEY not set. Please set it with: export DATALAB_API_KEY='your_key'"
    echo "Skipping Chandra OCR for now."
else
    # Run Chandra OCR
    echo "Running Chandra OCR..."
    # Assume chandra.py is configured for images in ../../../data/OmniDocBench/images/
    conda run -n olmocr python chandra.py
    echo "Chandra OCR complete. Check outputs for results."
fi

cd $ABS_PATH

# Evaluation
echo "=========================================="
echo "Starting Evaluation on OmniDocBench..."

conda create -n omnidocbench-eval python=3.10 -y
conda run -n omnidocbench-eval pip install -e .[omnidocbench] 
eval "$(conda shell.bash hook)"
conda activate omnidocbench-eval
echo "Evaluation environment setup complete."

echo "=========================================="
echo "EVALUATION PHASE"
echo "=========================================="

echo ""
echo "Running Evaluation on OmniDocBench dataset..."
echo "Language filter: $LANGUAGE_FILTER"
echo ""

cd $ABS_PATH/src/omnidocbench_evals/OmniDocBench

# ==========================================
# DEEPSEEK-OCR EVALUATION
# ==========================================
echo "Starting DeepSeek-OCR Evaluation..."
echo "Output directory: $ABS_PATH/outputs/deepseek_ocr/"

# Update config paths for DeepSeek results
sed -i "s|data_path: .*/OmniDocBench.json|data_path: $ABS_PATH/data/OmniDocBench/OmniDocBench.json|" configs/end2end.yaml
sed -i "s|data_path: output_results_markdown|data_path: $ABS_PATH/outputs/deepseek_ocr/|" configs/end2end.yaml

# Apply language filter to configuration
if [ "$LANGUAGE_FILTER" != "all" ]; then
    echo "  → Filtering to: $LANGUAGE_FILTER"
    sed -i "s|# filter:|filter:|" configs/end2end.yaml
    sed -i "s|#   language: english|  language: $LANGUAGE_FILTER|" configs/end2end.yaml
else
    echo "  → Including all languages"
    sed -i "s|filter:|# filter:|" configs/end2end.yaml
    sed -i "s|  language: english|#   language: english|" configs/end2end.yaml
    sed -i "s|  language: simplified_chinese|#   language: simplified_chinese|" configs/end2end.yaml
fi

# Execute DeepSeek evaluation
if [ -f "pdf_validation.py" ]; then
    echo "  → Running pdf_validation with DeepSeek-OCR results..."
    python pdf_validation.py --config configs/end2end.yaml --ocr-type deepseek_ocr --language "$LANGUAGE_FILTER"
    echo "✓ DeepSeek-OCR evaluation complete"
else
    echo "✗ pdf_validation.py not found. Please run evaluation manually."
fi

echo ""

# ==========================================
# OLMOCR2 EVALUATION
# ==========================================
echo "Starting OLMOCR2 Evaluation..."
echo "Output directory: $ABS_PATH/outputs/olmocr_workspace/markdown/"

# Update config paths for OLMOCR results
sed -i "s|data_path: .*/OmniDocBench.json|data_path: $ABS_PATH/data/OmniDocBench/OmniDocBench.json|" configs/end2end.yaml
sed -i "s|data_path: output_results_markdown|data_path: $ABS_PATH/outputs/olmocr_workspace/markdown/|" configs/end2end.yaml

# Apply language filter to configuration
if [ "$LANGUAGE_FILTER" != "all" ]; then
    echo "  → Filtering to: $LANGUAGE_FILTER"
    sed -i "s|# filter:|filter:|" configs/end2end.yaml
    sed -i "s|#   language: english|  language: $LANGUAGE_FILTER|" configs/end2end.yaml
else
    echo "  → Including all languages"
    sed -i "s|filter:|# filter:|" configs/end2end.yaml
    sed -i "s|  language: english|#   language: english|" configs/end2end.yaml
    sed -i "s|  language: simplified_chinese|#   language: simplified_chinese|" configs/end2end.yaml
fi

# Execute OLMOCR2 evaluation
if [ -f "pdf_validation.py" ]; then
    echo "  → Running pdf_validation with OLMOCR2 results..."
    python pdf_validation.py --config configs/end2end.yaml --ocr-type olmocr2 --language "$LANGUAGE_FILTER"
    echo "✓ OLMOCR2 evaluation complete"
else
    echo "✗ pdf_validation.py not found. Please run evaluation manually."
fi

echo ""


echo "=========================================="
echo "Setup and Evaluation Complete!"
echo "=========================================="
echo ""
echo "Evaluation results saved to:"
echo "  • Metric files: src/omnidocbench_evals/OmniDocBench/result/"
echo "  • Summary CSV: results_${LANGUAGE_FILTER}.csv"
echo ""

# ==========================================
# RESULTS SUMMARY
# ==========================================
echo ""
echo "=========================================="
echo "GENERATING RESULTS SUMMARY"
echo "=========================================="
echo ""
echo "Aggregating metrics from both OCR systems..."
echo "Language filter: $LANGUAGE_FILTER"
echo ""

# Navigate back to omnidocbench_evals root directory
cd $ABS_PATH

# Generate comprehensive results summary
python run_evaluation.py \
    --result-dir "$ABS_PATH/src/omnidocbench_evals/OmniDocBench/result" \
    --ocr-types deepseek olmocr2 \
    --language "$LANGUAGE_FILTER" \
    --match-method quick_match \
    --output "$ABS_PATH/results_${LANGUAGE_FILTER}.csv"

echo ""
echo "=========================================="
echo "✓ EVALUATION WORKFLOW COMPLETE!"
echo "=========================================="
echo ""
echo "Results Summary:"
echo "  • CSV File: results_${LANGUAGE_FILTER}.csv"
echo "  • Detailed Metrics: src/omnidocbench_evals/OmniDocBench/result/"
echo "  • Language Filter Applied: $LANGUAGE_FILTER"
echo ""