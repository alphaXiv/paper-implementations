# Grassmann Flow Paper Reproduction: Technical Report

## Paper Reference

**Title:** "Attention Is Not What You Need"
**arXiv:** 2512.19428
**Claim:** Grassmann flow layers achieve perplexity "within 10-15% of size-matched Transformers" on Wikitext-2

---

## Executive Summary

We attempted an exact reproduction of the paper's Wikitext-2 experiments. Our results show a **22.6% performance gap** between Grassmann and Transformer models, significantly exceeding the paper's claimed 10-15% gap.

---

## Methodology

### Dataset
- **Wikitext-2** (wikitext-2-raw-v1)
- Train: 9,343 chunks (256 tokens each)
- Validation: 965 chunks
- Test: 1,106 chunks
- Tokenizer: GPT2Tokenizer (vocab size: 50,257)

### Model Configurations

#### GrassmannGPT (Exact Paper Architecture)
```
Parameters: 17,695,168 (17.70M)
- model_dim: 256
- num_layers: 6
- reduced_dim: 32 (paper's r value)
- window_sizes: [1, 2, 4, 8, 12, 16]
- ff_dim: 1024 (4x model_dim)
- dropout: 0.1
- tie_weights: True
```

Key architectural details per paper:
1. Plucker coordinates: p_ij = z_t_i * z_{t-delta}_j - z_t_j * z_{t-delta}_i
2. L2 normalization of Plucker before projection
3. Blend gating: h_mix = alpha * h + (1-alpha) * g
4. Gate input: concatenate [h; g]
5. Order: Plucker -> L2 norm -> Proj -> Avg -> Gate blend -> LayerNorm

#### Size-Matched Transformer Baseline
```
Parameters: 17,670,400 (17.67M)
- model_dim: 256
- num_layers: 6
- num_heads: 8
- ff_dim: 1024 (4x model_dim)
- dropout: 0.1
- tie_weights: True
```

### Training Configuration
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Scheduler: CosineAnnealingLR
- Epochs: 20
- Batch size: 32
- Gradient clipping: 1.0
- Hardware: NVIDIA H100 SXM5 80GB

---

## Results

### Training Curves

#### Grassmann Model
| Epoch | Train Loss | Val Loss | Val PPL |
|-------|------------|----------|---------|
| 1     | 7.2473     | 6.4292   | 619.66  |
| 5     | 5.2780     | 5.6617   | 287.65  |
| 10    | 4.7262     | 5.4898   | 242.21  |
| 15    | 4.5029     | 5.4640   | 236.05  |
| 20    | 4.4482     | 5.4654   | 236.36  |

#### Transformer Model
| Epoch | Train Loss | Val Loss | Val PPL |
|-------|------------|----------|---------|
| 1     | 7.1584     | 6.3960   | 599.42  |
| 5     | 5.2462     | 5.5120   | 247.64  |
| 10    | 4.5300     | 5.2681   | 194.04  |
| 13    | 4.3043     | 5.2472   | 190.03  |
| 20    | 4.1350     | 5.2630   | 193.05  |

### Final Comparison

| Model       | Parameters | Best Val PPL | Test PPL |
|-------------|-----------|--------------|----------|
| Grassmann   | 17.70M    | 236.05       | 242.94   |
| Transformer | 17.67M    | 190.03       | 198.17   |

### Gap Analysis

```
Test PPL Ratio: 242.94 / 198.17 = 1.226
Gap: 22.6%
Paper Claim: 10-15%
Discrepancy: +7.6% to +12.6% worse than claimed
```

---

## Observations

### 1. Convergence Behavior
- Transformer converges faster and to a lower loss
- Grassmann shows diminishing returns after epoch 15
- Both models exhibit slight overfitting in final epochs

### 2. Training Dynamics
- Grassmann: ~53k tokens/sec on H100
- Transformer: Similar throughput (attention is efficient at 256 seq len)
- Total training time: ~30 minutes for both models

### 3. Architecture Differences
- Grassmann lacks explicit attention mechanism
- Uses geometric (Plucker) coordinates for token mixing
- Gating mechanism blends original with geometric features

---

## Potential Sources of Discrepancy

### Hyperparameters Not Specified in Paper
1. **Learning rate schedule details** - warmup steps, minimum LR
2. **Weight initialization** - paper doesn't specify init scheme
3. **Layer-specific window sizes** - paper may use different windows per layer
4. **Dropout placement** - exact positions not fully specified

### Implementation Ambiguities
1. **Normalization order** - pre-norm vs post-norm variations
2. **Gating activation** - sigmoid range and initialization
3. **Residual connections** - exact formulation
4. **FFN structure** - GELU vs other activations

### Experimental Setup
1. **Tokenization** - GPT2 vs custom tokenizer
2. **Data preprocessing** - chunk boundaries, padding
3. **Evaluation protocol** - sliding window vs chunk-based
4. **Random seeds** - reproducibility factors

---

## Conclusions

1. **Paper claims not reproduced**: Our 22.6% gap significantly exceeds the claimed 10-15%

2. **Grassmann underperforms**: Even with exact paper architecture (v4), the geometric approach falls short of standard attention

3. **Limited practical utility**: At this performance gap, Grassmann flows are not competitive for language modeling

4. **Possible explanations**:
   - Missing hyperparameter details in paper
   - Implementation differences in gating or normalization
   - Dataset/tokenization variations
   - The paper's results may have been optimistically reported

---

## Recommendations for Blog Post

1. **Be factual**: Report our 22.6% gap vs paper's 10-15% claim
2. **Acknowledge limitations**: We may have missed implementation details
3. **Context**: Compare to other attention alternatives (Mamba, RWKV, Linear Attention)
4. **Conclusion**: Grassmann flows remain an interesting theoretical contribution but lack practical competitiveness

---

## CUDA Kernel Optimization

Custom CUDA kernels were implemented for the Plucker coordinate computation.

### Performance Results (H100 80GB)

| Metric | PyTorch | CUDA | Speedup |
|--------|---------|------|---------|
| Mixing layer forward | 1.59 ms | 0.35 ms | 4.6x |
| Full model inference | 9.16 ms | 4.53 ms | 2.0x |
| Inference throughput | 0.45M tok/s | 0.90M tok/s | 2.0x |

### Correctness Verification

- 100 training iterations compared
- Loss difference: 0.0001-0.0002 (numerical precision)
- Implementation verified correct

---

## Files and Artifacts

- Training script: `train_wikitext2.py`
- Model implementation: `src/models/grassmann.py`
- CUDA kernels: `src/cuda/grassmann_kernels.cu`
- CUDA wrapper: `src/cuda/grassmann_fused.py`
- CUDA tests: `test_cuda_kernels.py`
- Results: `outputs/wikitext2_reproduction/results.json`
- Model checkpoints: `outputs/wikitext2_reproduction/{grassmann,transformer}_best.pt`

---

## Reproducibility

To reproduce these results:

```bash
# On machine with GPU
python train_wikitext2.py --model both --epochs 20 --model-dim 256 --num-layers 6
```

Hardware used: NVIDIA H100 SXM5 80GB (Voltage Park)

---

## Raw Results JSON

```json
{
  "grassmann": {
    "num_params": 17695168,
    "best_val_loss": 5.4640497691890735,
    "best_val_ppl": 236.05145263671875,
    "test_loss": 5.4928154108968394,
    "test_ppl": 242.94024658203125
  },
  "transformer": {
    "num_params": 17670400,
    "best_val_loss": 5.247161563813995,
    "best_val_ppl": 190.02609252929688,
    "test_loss": 5.289115955773573,
    "test_ppl": 198.16815185546875
  }
}
```
