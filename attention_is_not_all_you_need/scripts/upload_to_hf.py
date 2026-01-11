#!/usr/bin/env python3
"""
Upload best model checkpoints to Hugging Face Hub.
Maintains the same folder structure as outputs directory.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import json

# Configuration
HF_USERNAME = "alphaXiv"  # Change this to your HF username
REPO_NAME = "attention-is-not-all-you-need-models"
OUTPUTS_DIR = Path("/home/ubuntu/central-texas-storage/paper-implementations/attention_is_not_all_you_need/outputs/2026-01-10_23-46-12")

def upload_models():
    """Upload all best.pt models and result files to HuggingFace."""
    
    api = HfApi()
    
    # Create repository (will skip if exists)
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"⚠️  Repository creation note: {e}")
        print("Proceeding with upload...")
    
    # Find all best.pt files
    model_dirs = [
        "grassmann_snli",
        "grassmann_wikitext_L128_N12",
        "grassmann_wikitext_L128_N6",
        "grassmann_wikitext_L256_N12",
        "grassmann_wikitext_L256_N6",
        "transformer_snli",
        "transformer_wikitext_L128_N12",
        "transformer_wikitext_L128_N6",
        "transformer_wikitext_L256_N12",
        "transformer_wikitext_L256_N6",
    ]
    
    uploaded_files = []
    
    for model_dir in model_dirs:
        dir_path = OUTPUTS_DIR / model_dir
        
        if not dir_path.exists():
            print(f"⚠️  Directory not found: {model_dir}")
            continue
        
        # Upload best.pt checkpoint
        checkpoint_path = dir_path / "checkpoints" / "best.pt"
        if checkpoint_path.exists():
            path_in_repo = f"{model_dir}/checkpoints/best.pt"
            print(f"📤 Uploading {checkpoint_path.name} from {model_dir}...")
            
            try:
                api.upload_file(
                    path_or_fileobj=str(checkpoint_path),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="model",
                )
                uploaded_files.append(path_in_repo)
                print(f"   ✅ Uploaded: {path_in_repo}")
            except Exception as e:
                print(f"   ❌ Failed to upload {path_in_repo}: {e}")
        
        # Upload result JSON files
        json_files = list(dir_path.glob("*.json"))
        for json_file in json_files:
            if json_file.name.startswith("wandb"):
                continue
            
            path_in_repo = f"{model_dir}/{json_file.name}"
            print(f"📤 Uploading {json_file.name} from {model_dir}...")
            
            try:
                api.upload_file(
                    path_or_fileobj=str(json_file),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="model",
                )
                uploaded_files.append(path_in_repo)
                print(f"   ✅ Uploaded: {path_in_repo}")
            except Exception as e:
                print(f"   ❌ Failed to upload {path_in_repo}: {e}")
    
    # Create a README for the repository
    readme_content = f"""---
license: mit
tags:
- grassmann-manifolds
- attention-alternative
- language-modeling
- natural-language-inference
- pytorch
library_name: pytorch
---

# Attention Is Not All You Need - Model Checkpoints

This repository contains the best model checkpoints from the reproduction of "Attention Is Not What You Need" paper.

## Models Included

### Wikitext-2 Language Modeling
- **Transformer Models**: L128/L256 with N6/N12 layers
- **Grassmann Models**: L128/L256 with N6/N12 layers

### SNLI Natural Language Inference
- **Transformer Model**: Classification head trained from scratch
- **Grassmann Model**: Classification head trained from scratch

## Results Summary

### Wikitext-2 (Best Validation PPL)
- **Best Transformer**: L=256, N=12 → 168.68 PPL
- **Best Grassmann**: L=128, N=12 → 244.61 PPL
- **Gap**: 45.0% (Grassmann underperforms)

### SNLI (Test Accuracy)
- **Grassmann**: 71.25% accuracy
- **Transformer**: 66.71% accuracy
- **Gap**: +4.54% (Grassmann outperforms!)

## Repository Structure

```
├── grassmann_snli/
│   ├── checkpoints/best.pt
│   ├── snli_test_results.json
│   └── snli_validation_results.json
├── grassmann_wikitext_L128_N6/
│   ├── checkpoints/best.pt
│   ├── results.json
│   └── wikitext_validation_results.json
├── grassmann_wikitext_L128_N12/
│   ├── checkpoints/best.pt
│   ├── results.json
│   └── wikitext_validation_results.json
├── grassmann_wikitext_L256_N6/
│   ├── checkpoints/best.pt
│   ├── results.json
│   └── wikitext_validation_results.json
├── grassmann_wikitext_L256_N12/
│   ├── checkpoints/best.pt
│   ├── results.json
│   └── wikitext_validation_results.json
├── transformer_snli/
│   ├── checkpoints/best.pt
│   ├── snli_test_results.json
│   └── snli_validation_results.json
├── transformer_wikitext_L128_N6/
│   ├── checkpoints/best.pt
│   ├── results.json
│   └── wikitext_validation_results.json
├── transformer_wikitext_L128_N12/
│   ├── checkpoints/best.pt
│   ├── results.json
│   └── wikitext_validation_results.json
├── transformer_wikitext_L256_N6/
│   ├── checkpoints/best.pt
│   ├── results.json
│   └── wikitext_validation_results.json
└── transformer_wikitext_L256_N12/
    ├── checkpoints/best.pt
    ├── results.json
    └── wikitext_validation_results.json
```

## Loading Models

```python
import torch

# Load a checkpoint
checkpoint = torch.load("grassmann_wikitext_L256_N12/checkpoints/best.pt")

# Access model state
model_state = checkpoint['model_state_dict']
epoch = checkpoint['epoch']
val_loss = checkpoint['val_loss']

print(f"Epoch: {{epoch}}, Val Loss: {{val_loss}}")
```

## Citation

If you use these models, please cite the original paper reproduction:

```bibtex
@misc{{attn-is-not-all-you-need-reproduction,
  title={{Reproduction of "Attention Is Not What You Need"}},
  author={{alphaXiv}},
  year={{2026}},
  url={{https://github.com/alphaXiv/paper-implementations}}
}}
```

## Hardware

All models trained on:
- **GPU**: NVIDIA H100 SXM5 80GB
- **Platform**: Lambda Labs, Lambda Stack 22.04

## License

MIT License
"""
    
    readme_path = Path("/tmp/README.md")
    readme_path.write_text(readme_content)
    
    try:
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"\n✅ Uploaded README.md")
    except Exception as e:
        print(f"\n❌ Failed to upload README: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("UPLOAD SUMMARY")
    print("="*60)
    print(f"Repository: https://huggingface.co/{repo_id}")
    print(f"Total files uploaded: {len(uploaded_files)}")
    print("\nUploaded files:")
    for f in sorted(uploaded_files):
        print(f"  ✓ {f}")
    print("="*60)

if __name__ == "__main__":
    print("Starting upload to Hugging Face Hub...")
    print(f"Source directory: {OUTPUTS_DIR}")
    print()
    
    # Check if logged in
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face!")
        print("Please run: huggingface-cli login")
        print("Or set HF_TOKEN environment variable")
        exit(1)
    
    print()
    upload_models()
