# DeepSeek-OCR vs OLMOCR-2 Evaluation on OmniDocBench

This repository provides a comparative evaluation of **DeepSeek-OCR** and **OLMOCR-2** on the **OmniDocBench** benchmark. The evaluation assesses document parsing capabilities across text, formulas, tables, and reading order.

## Overview

- **DeepSeek-OCR**: A vLLM-based multimodal pipeline for document understanding.
- **OLMOCR-2**: An efficient OCR system using open visual language models.
- **OmniDocBench**: A comprehensive benchmark with 1,355 annotated PDF pages covering diverse document types.

## Setup

### 1. DeepSeek-OCR Setup


## Install
>Our environment is cuda11.8+torch2.6.0.
1. Clone this repository and navigate to the DeepSeek-OCR folder
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd src/
```
```

2. Conda
```Shell
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```
3. Packages

- download the vllm-0.8.5 [whl](https://github.com/vllm-project/vllm/releases/tag/v0.8.5) 

```Shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```
**Note:** if you want vLLM and transformers codes to run in the same environment, you don't need to worry about this installation error like: vllm 0.8.5+cu118 requires transformers>=4.51.1


Follow the installation guide in case of any doubts in [`src/DeepSeek-OCR-master/README.md`](DeepSeek-OCR-master/README.md).


CAUTION: DO NOT install olmOCR2 and DeepSeek-OCR in the same conda environment to avoid package conflicts.

### 2. OLMOCR Setup


```bash

conda create -n olmocr python=3.11
conda activate olmocr

# For actually converting the files with your own GPU
pip install olmocr[gpu]  --extra-index-url https://download.pytorch.org/whl/cu128

# Recommended: Install flash infer for faster inference on GPU
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl

``` 
### 3. Chandra OCR Setup

Quick usage
1. install chandra-ocr package

   ```bash
   pip install chandra-ocr
   ```

Follow the setup instructions in [`src/olmocr/README.md`](olmocr/README.md).


## Get the data

We used the HuggingFace version and based all our evals on it.

Can be found at [link](https://huggingface.co/datasets/opendatalab/OmniDocBench)


For olmOCR2, convert the images to PDFs using the following

```bash

python OmniDocBench/utils/image_to_pdf.py #(make sure to set the input directory inside the script)

```

Our outputs can be found in the 'outputs' folder.


## Running the Models

Make sure to follow the setup instructions for each of the models before running their respective inference.

### Generate Outputs from DeepSeek-OCR

1. Navigate to the DeepSeek-OCR directory:
   ```bash
   cd src/DeepSeek-OCR-master/DeepSeek-OCR-vllm
   ```

2. Configure paths in `config.py`:
   - Set `INPUT_PATH` to the OmniDocBench images directory (e.g., `../../OmniDocBench/images/`)
   - Set `OUTPUT_PATH` to a directory for output .md files (e.g., `../../outputs/deepseek_ocr/`)

3. Run inference on images:
   ```bash
   python run_dpsk_ocr_eval_batch.py
   ```

   This will process all images and generate corresponding `.md` files in the output directory. Remember to use 'cleaned' .md files for evaluation, which can be found in `./DeepSeek-OCR-OmniDocBench/outputs/markdowns_for_dpsk_ocr/results_dpsk-cleaned` that we generated.

### Generate Outputs from OLMOCR-2

1. Navigate to the olmocr directory:
   ```bash
   cd olmocr
   ```

2. Run inference on PDFs:
   ```bash
   python -m olmocr.pipeline ./localworkspace --markdown --pdfs tests/gnarly_pdfs/*.pdf #put your pdf path here
   ```

   Replace `tests/gnarly_pdfs/` with a workspace directory, that includes your pdf files.

   The `--markdown` flag ensures `.md` files are generated in the workspace's `markdown/` subdirectory.

### 3. Generate outputs from Chandra OCR

DATALAB_API_KEY is needed to run chandra OCR.

Set the environment variable before running the script:

```bash
export DATALAB_API_KEY="your_api_key_here"
```

Make sure to give the directory of images or pdfs inside the chandra.py script or modify the script to take input arguments.


1. Navigate to the chandra-ocr directory:
   ```bash
   cd src/chandra-ocr
   ```
2. Run the chandra.py script (example):
   ```bash
   python chandra.py
   ```

## Evaluation

Use OmniDocBench's evaluation scripts to compare the generated outputs.

### End-to-End Evaluation (end2end)

1. Configure `OmniDocBench/configs/end2end.yaml`:
   - Set `ground_truth.data_path` to `OmniDocBench/OmniDocBench.json`
   - Set `prediction.data_path` to the directory containing model outputs (e.g., `outputs/deepseek_ocr/` or `olmocr_workspace/markdown/`)

2. Run evaluation:
   ```bash
   cd OmniDocBench
   python pdf_validation.py --config configs/end2end.yaml
   ```


## Results

After evaluation, results are stored in `OmniDocBench/result/`. Use the notebooks in `OmniDocBench/tools/` to generate comparison tables and visualizations.

You can find one of our results for one of our runs in `outputs/results_chandra_ocr` folder for eg! Remember we used Edit_dist for formula evaluation instead of CDM .

Key metrics include:
- Text accuracy (normalized edit distance)
- Formula accuracy (Edit dist score)
- Table TEDS score
- Reading order accuracy
- Overall score: ((1 - text_edit) × 100 + table_teds + (1 - edit_distance) × 100) / 3


See [`REPORT.md`](REPORT.md) for detailed results and visualizations.

