---
language: en
license: mit
tags:
- trm
- recursive-reasoning
- arc-agi
- abstract-reasoning
- pytorch
- huggingface
datasets:
- ARC-AGI
metrics:
- pass@2
widget:
- text: "Sample ARC task here"
---

# TRM Model for ARC-AGI-1

## Model Description

This is a Tiny Recursive Model (TRM) fine-tuned for solving Abstract Reasoning Challenge (ARC-AGI) tasks. The model performs abstract reasoning to predict output grids from input grids.

- **Developed by:** alphaXiv
- **Model type:** TRM-Attention
- **Language(s) (NLP):** N/A (grid-based reasoning)
- **License:** MIT
- **Finetuned from model:** Custom TRM architecture

## Intended Use

### Primary Use

This model is designed to solve ARC-AGI tasks by predicting the correct output grid transformation based on input grid patterns.

### Out-of-Scope Use

Not intended for general NLP tasks, image generation, or other reasoning domains.

## Limitations and Bias

- Trained only on ARC-AGI training and evaluation sets
- May not generalize to novel abstract reasoning tasks
- Performance limited by training data diversity

## Training Data

The model was trained on the ARC-AGI dataset, which includes:
- Input-output grid pairs
- Various transformation patterns
- Training and evaluation splits

## Evaluation Results

| Metric | Claimed | Achieved |
|--------|---------|----------|
| Pass@2 | 44.6% | 43.00% ± 0.16% |

Results from independent reproduction study.

## Repository

https://github.com/alphaXiv/TinyRecursiveModels
