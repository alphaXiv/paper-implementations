#!/bin/bash

# OmniDocBench-Evals Speedrun Script
# Sets up and runs evaluation for DeepSeek-OCR, OLMOCR-2, and Chandra OCR on OmniDocBench

set -e

echo "=========================================="
echo "OmniDocBench-Evals Setup and Evaluation"
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

# Download OmniDocBench dataset
echo "Downloading OmniDocBench dataset..."
conda run -n base pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='opendatalab/OmniDocBench', repo_type='dataset', local_dir='data/OmniDocBench')"
echo "Dataset downloaded to data/OmniDocBench"

# Convert images to PDFs for OLMOCR
echo "Converting images to PDFs for OLMOCR..."
# Install basic dependencies
conda run -n base pip install PyMuPDF tqdm 

# Get absolute path
ABS_PATH=$(pwd)

# Assume the images are in data/OmniDocBench/images/
# Edit the script to set input_dir and output_dir
# sed -i "s|input_directory = .*|input_directory = \"$ABS_PATH/data/OmniDocBench/images/\"|" src/omnidocbench-evals/OmniDocBench/tools/image_to_pdf.py
# sed -i "s|output_directory = .*|output_directory = \"$ABS_PATH/data/OmniDocBench/pdfs/\"|" src/omnidocbench-evals/OmniDocBench/tools/image_to_pdf.py
# python src/omnidocbench-evals/OmniDocBench/tools/image_to_pdf.py
# echo "PDFs created."

# Setup DeepSeek-OCR
echo "Setting up DeepSeek-OCR..."

conda create -n deepseek-ocr python=3.12.9 -y
conda run -n deepseek-ocr pip install -e .[deepseek-ocr] --extra-index-url https://download.pytorch.org/whl/cu118
conda run -n deepseek-ocr pip install flash_attn==2.7.3 --no-build-isolation
cd src/omnidocbench-evals/DeepSeek-OCR-master/DeepSeek-OCR-vllm
# # Configure config.py
# sed -i "s|INPUT_PATH = 'data/OmniDocBench/images/'|INPUT_PATH = '../../../../data/OmniDocBench/images/'|" config.py
# sed -i "s|OUTPUT_PATH = 'outputs/deepseek_ocr/'|OUTPUT_PATH = '../../../../outputs/deepseek_ocr/'|" config.py
# echo "DeepSeek-OCR setup complete."

# # Run DeepSeek-OCR inference
# echo "Running DeepSeek-OCR inference..."
# conda run -n deepseek-ocr python run_dpsk_ocr_eval_batch.py
# echo "DeepSeek-OCR inference complete. Outputs in outputs/deepseek_ocr/"

echo "Cleaning up the outputs/deepseek_ocr/ directory..."
# rm ../../../../outputs/deepseek_ocr/*.png

echo "Cleaning complete. Noe setting up olmOCR2.."

cd ../../../../src/omnidocbench-evals/olmocr

# Setup OLMOCR
echo "Setting up OLMOCR..."
sudo apt-get update
sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
conda create -n olmocr python=3.11 -y
conda run -n olmocr pip install -e .[olmocr] --extra-index-url https://download.pytorch.org/whl/cu128
conda run -n olmocr pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128

echo "OLMOCR setup complete."

# Run OLMOCR inference
echo "Running OLMOCR inference..."
# Assume PDFs are in data/OmniDocBench/pdfs/ (created by image_to_pdf.py)
mkdir -p ../../../outputs/olmocr_workspace
# Create a text file listing all PDFs
find ../../../data/OmniDocBench/pdfs/ -name "*.pdf" -type f > ../../../outputs/olmocr_workspace/pdf_list.txt
echo "Found $(wc -l < ../../../outputs/olmocr_workspace/pdf_list.txt) PDF files to process"
# Use eval to properly activate the conda environment so vllm is in PATH
eval "$(conda shell.bash hook)"
conda activate olmocr
python -m olmocr.pipeline ../../../outputs/olmocr_workspace --markdown --pdfs ../../../outputs/olmocr_workspace/pdf_list.txt
conda deactivate
echo "OLMOCR inference complete. Converting JSONL to markdown..."
# Convert JSONL results to markdown files
python ../../../src/omnidocbench-evals/olmocr/convert_jsonl_to_markdown.py
echo "Markdown files created in outputs/olmocr_workspace/markdown/"

cd ../../..

# Setup Chandra OCR
echo "Setting up Chandra OCR..."
# Reuse olmocr env to avoid conflicts
conda run -n olmocr pip install -e .[chandra-ocr]
cd src/omnidocbench-evals/chandra-ocr
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

cd ../../..

# Evaluation
echo "Running Evaluation..."

echo "DeepSeek-OCR Evaluation..."

cd src/omnidocbench-evals/OmniDocBench
# Edit config
sed -i "s|data_path: .*/OmniDocBench.json|data_path: ../../../data/OmniDocBench/OmniDocBench.json|" configs/end2end.yaml
sed -i "s|data_path: output_results_markdown|data_path: ../../outputs/deepseek_ocr/|" configs/end2end.yaml  # For DeepSeek, use cleaned if available
# Run evaluation (assuming pdf_validation.py exists or use alternative)
if [ -f "pdf_validation.py" ]; then
    conda run -n deepseek-ocr python pdf_validation.py --config configs/end2end.yaml
else
    echo "pdf_validation.py not found. Please run evaluation manually using the notebooks in tools/"
fi

echo "OLMOCR Evaluation..."
# Edit config
sed -i "s|data_path: .*/OmniDocBench.json|data_path: ../../../data/OmniDocBench/OmniDocBench.json|" configs/end2end.yaml
sed -i "s|data_path: output_results_markdown|data_path: ../../outputs/olmocr_workspace/markdown/|" configs/end2end.yaml  # For OLMOCR
# Run evaluation (assuming pdf_validation.py exists or use alternative)
if [ -f "pdf_validation.py" ]; then
    conda run -n olmocr python pdf_validation.py --config configs/end2end.yaml
else
    echo "pdf_validation.py not found. Please run evaluation manually using the notebooks in tools/"
fi

echo "=========================================="
echo "Setup and Evaluation Complete!"
echo "Check outputs/ for results and use notebooks in OmniDocBench/tools/ for detailed analysis."
echo "=========================================="