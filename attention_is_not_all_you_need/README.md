# Attention Is Not All You Need

An independent reproduction study of "Attention Is Not What You Need" (arXiv 2512.19428).

## Summary

This repository contains a reproduction of Grassmann flow layers for sequence modeling. The original paper claims performance "within 10-15% of size-matched Transformers" on Wikitext-2. Our reproduction shows a **22.6% gap** - significantly larger than claimed.

## Key Results

| Model | Parameters | Test PPL |
|-------|------------|----------|
| Grassmann (paper arch) | 17.70M | 242.94 |
| Transformer | 17.67M | 198.17 |

**Gap: 22.6%** (vs claimed 10-15%)

## CUDA Optimization

Custom CUDA kernels provide **2x inference speedup**:

| Metric | PyTorch | CUDA | Speedup |
|--------|---------|------|---------|
| Full model inference | 9.16 ms | 4.53 ms | 2.0x |

## Blog Post

Full analysis and discussion: [blog.md](blog.md)

## Quick Start

```bash
# Install dependencies
pip install torch datasets transformers tqdm

# Run reproduction
python train_wikitext2.py --model both --epochs 20

# Build CUDA kernels (optional)
cd src/cuda && python setup.py install
```

## Files

- `train_wikitext2.py` - Training script
- `src/models/grassmann.py` - Paper-exact implementation
- `src/cuda/` - CUDA kernel implementation
- `blog.md` - Full reproduction report
- `technical.md` - Technical details

