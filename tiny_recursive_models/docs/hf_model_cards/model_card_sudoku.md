---
language: en
license: mit
tags:
- trm
- recursive-reasoning
- sudoku
- pytorch
- huggingface
datasets:
- custom
metrics:
- accuracy
widget:
- text: "Sample sudoku puzzle here"
---

# TRM Model for Sudoku Solving

## Model Description

This is a Tiny Recursive Model (TRM) fine-tuned for solving Sudoku puzzles. The model uses recursive reasoning to fill in missing numbers in Sudoku grids.

- **Developed by:** alphaXiv
- **Model type:** TRM-MLP
- **Language(s) (NLP):** N/A (grid-based reasoning)
- **License:** MIT
- **Finetuned from model:** Custom TRM architecture

## Intended Use

### Primary Use

This model is designed to solve Sudoku puzzles by predicting the correct numbers for empty cells in standard 9x9 Sudoku grids.

### Out-of-Scope Use

Not intended for general NLP tasks, image processing, or other puzzle types.

## Limitations and Bias

- Trained only on standard 9x9 Sudoku puzzles
- May not handle non-standard Sudoku variants
- Performance depends on puzzle difficulty

## Training Data

The model was trained on a dataset of Sudoku puzzles with extreme difficulty levels. The dataset includes:
- Partially filled 9x9 grids
- Correct solutions
- Difficulty ratings

## Evaluation Results

| Variant | Metric | Claimed | Achieved |
|---------|--------|---------|----------|
| TRM-MLP | Accuracy | 87.4% | 79.37% ± 0.12% |
| TRM-Attention | Accuracy | 74.7% | 73.66% ± 0.13% |

Results from independent reproduction study.

## Repository

https://github.com/alphaXiv/TinyRecursiveModels