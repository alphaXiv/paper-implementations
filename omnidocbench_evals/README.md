# DeepSeek-OCR vs OLMOCR-2 Evaluation on OmniDocBench

This repository provides a comparative evaluation of **DeepSeek-OCR** and **OLMOCR-2** on the **OmniDocBench** benchmark. The evaluation assesses document parsing capabilities across text, formulas, tables, and reading order.

## Overview

- **DeepSeek-OCR**: A vLLM-based multimodal pipeline for document understanding.
- **OLMOCR-2**: An efficient OCR system using open visual language models.
- **OmniDocBench**: A comprehensive benchmark with 1,355 annotated PDF pages covering diverse document types.

## Quick Start

The easiest way to set up and run the complete evaluation is using the automated speedrun script:

```bash
./speedrun.sh [language_filter]
```

**Options:**
- `all` - Evaluate on all languages (default)
- `english` - Evaluate on English documents only
- `simplified_chinese` - Evaluate on simplified Chinese documents only

**Example:**
```bash
./speedrun.sh all
```

This script will automatically:
1. Download the OmniDocBench dataset
2. Set up all three OCR environments (DeepSeek-OCR, OLMOCR-2, Chandra OCR)
3. Run inference on all models
4. Execute comprehensive evaluation
5. Generate comparison results

### Environment Requirements

- CUDA 11.8+ with torch 2.6.0+
- Conda/Miniconda installed
- At least 60GB free disk space for models and data
- For Chandra OCR: `DATALAB_API_KEY` environment variable (optional)
## Results

After evaluation, results are stored in `OmniDocBench/result/` and a summary CSV is generated at `results_${LANGUAGE_FILTER}.csv`.

The speedrun script will automatically generate:
- **CSV Summary**: `results_all.csv` (or `results_english.csv`, `results_simplified_chinese.csv` based on language filter)
- **Detailed Metrics**: `src/omnidocbench_evals/OmniDocBench/result/`
- **Model Outputs**: 
  - DeepSeek-OCR: `outputs/deepseek_ocr/`
  - OLMOCR-2: `outputs/olmocr_workspace/markdown/`
  - Chandra OCR: `outputs/chandra_ocr/`

### Key Metrics

- **Text Accuracy**: Normalized edit distance
- **Formula Accuracy**: Edit distance score
- **Table TEDS Score**: Table structure evaluation
- **Reading Order Accuracy**: Document flow evaluation
- **Overall Score**: `((1 - text_edit) × 100 + table_teds + (1 - edit_distance) × 100) / 3`

See [`REPORT.md`](REPORT.md) for detailed results and visualizations from our evaluation runs.

## Troubleshooting

### CUDA/Memory Issues
- Ensure CUDA 11.8+ is installed: `nvidia-smi`
- Check available GPU memory: `nvidia-smi`
- Reduce batch size if out of memory

### Dataset Download Issues
- Set HuggingFace token: `export HF_TOKEN="your_token_here"`
- Check internet connection and rate limits

### Environment Conflicts
- Each OCR system runs in a separate conda environment to avoid conflicts
- Do not mix environments manually

## Data

The dataset is automatically downloaded from [OmniDocBench on HuggingFace](https://huggingface.co/datasets/opendatalab/OmniDocBench) to `data/OmniDocBench/` when running the speedrun script.

