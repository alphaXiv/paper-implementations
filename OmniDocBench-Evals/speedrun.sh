#!/bin/bash

# OmniDocBench-Evals Speedrun Script
# Sets up and runs evaluation for DeepSeek-OCR, OLMOCR-2, and Chandra OCR on OmniDocBench

set -e

echo "=========================================="
echo "OmniDocBench-Evals Setup and Evaluation"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    source ~/.bashrc
    echo "Conda installed and initialized."
else
    echo "Conda is already installed."
fi

# Ensure conda is available
export PATH="$HOME/miniconda/bin:$PATH"

# Create data directory
mkdir -p data

# Download OmniDocBench dataset
echo "Downloading OmniDocBench dataset..."
conda run -n base pip install huggingface_hub
conda run -n base python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='opendatalab/OmniDocBench', repo_type='dataset', local_dir='data/OmniDocBench')"
echo "Dataset downloaded to data/OmniDocBench"

# Convert images to PDFs for OLMOCR
echo "Converting images to PDFs for OLMOCR..."
# Assume the images are in data/OmniDocBench/images/
# Edit the script to set input_dir
sed -i 's|input_dir = .*|input_dir = "../../../data/OmniDocBench/images/"|' src/omnidocbench-evals/OmniDocBench/tools/image_to_pdf.py
conda run -n base python src/omnidocbench-evals/OmniDocBench/tools/image_to_pdf.py
echo "PDFs created."

# Setup DeepSeek-OCR
echo "Setting up DeepSeek-OCR..."
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
cd src/omnidocbench-evals/DeepSeek-OCR-master
pip install -e .[deepseek-ocr] --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd DeepSeek-OCR-vllm
# Configure config.py
sed -i "s|INPUT_PATH = ''|INPUT_PATH = '../../../data/OmniDocBench/images/'|" config.py
sed -i "s|OUTPUT_PATH = ''|OUTPUT_PATH = '../../../outputs/deepseek_ocr/'|" config.py
echo "DeepSeek-OCR setup complete."

# Run DeepSeek-OCR inference
echo "Running DeepSeek-OCR inference..."
python run_dpsk_ocr_eval_batch.py
echo "DeepSeek-OCR inference complete. Outputs in outputs/deepseek_ocr/"

conda deactivate
cd ../../../../..

# Setup OLMOCR
echo "Setting up OLMOCR..."
conda create -n olmocr python=3.11 -y
conda activate olmocr
cd src/omnidocbench-evals/olmocr
pip install -e .[olmocr] --extra-index-url https://download.pytorch.org/whl/cu128
echo "OLMOCR setup complete."

# Run OLMOCR inference
echo "Running OLMOCR inference..."
# Assume PDFs are in data/OmniDocBench/pdfs/ (created by image_to_pdf.py)
mkdir -p ../../../outputs/olmocr_workspace
python -m olmocr.pipeline ../../../outputs/olmocr_workspace --markdown --pdfs ../../../data/OmniDocBench/pdfs/*.pdf
echo "OLMOCR inference complete. Outputs in outputs/olmocr_workspace/markdown/"

conda deactivate
cd ../../..

# Setup Chandra OCR
echo "Setting up Chandra OCR..."
# Use a separate env or base
conda activate olmocr  # Reuse env to avoid conflicts
cd src/omnidocbench-evals/chandra-ocr
pip install -e .[chandra-ocr]
echo "Chandra OCR setup complete."

# Check for API key
if [ -z "$DATALAB_API_KEY" ]; then
    echo "WARNING: DATALAB_API_KEY not set. Please set it with: export DATALAB_API_KEY='your_key'"
    echo "Skipping Chandra OCR for now."
else
    # Run Chandra OCR
    echo "Running Chandra OCR..."
    # Assume chandra.py is configured for images in ../../../data/OmniDocBench/images/
    python chandra.py
    echo "Chandra OCR complete. Check outputs for results."
fi

conda deactivate
cd ../../..

# Evaluation
echo "Running Evaluation..."
cd src/omnidocbench-evals/OmniDocBench
# Edit config
sed -i "s|data_path: .*/OmniDocBench.json|data_path: ../../../data/OmniDocBench/OmniDocBench.json|" configs/end2end.yaml
sed -i "s|data_path: output_results_markdown|data_path: ../../outputs/deepseek_ocr/|" configs/end2end.yaml  # For DeepSeek, use cleaned if available
# Run evaluation (assuming pdf_validation.py exists or use alternative)
if [ -f "pdf_validation.py" ]; then
    python pdf_validation.py --config configs/end2end.yaml
else
    echo "pdf_validation.py not found. Please run evaluation manually using the notebooks in tools/"
fi

echo "=========================================="
echo "Setup and Evaluation Complete!"
echo "Check outputs/ for results and use notebooks in OmniDocBench/tools/ for detailed analysis."
echo "=========================================="