# Grassmann Flows for Sequence Modeling: Reproduction Study

An independent reproduction of "Attention Is Not What You Need" (arXiv 2512.19428), which proposes replacing transformer self-attention with Grassmann manifold-based geometric operations using Plucker coordinates.

**Author:** Elliot Arledge  
**Hardware:** NVIDIA H100 SXM5 80GB (1x GPU, Lambda Labs, Lambda Stack Ubuntu 22.04)  
**Infrastructure:** Used for all training and evaluation experiments

## 🎯 TL;DR

- ❌ **Wikitext-2 Gap:** Grassmann flows show 43-49% higher validation perplexity vs Transformers (paper claimed 10-11%)
- ✅ **SNLI Surprise:** Grassmann **outperforms** Transformer by 3.68% (66.47% vs 62.79%) when trained from scratch
- 📊 **Best Validation Results:** Transformer L256 N12 achieves 169.25 Val PPL vs Grassmann L256 N12 at 241.60 Val PPL
- 📉 **Discrepancy:** Wikitext gap is **4x larger** than paper's reported results
- 🔍 **Paper vs Ours:** Paper uses pre-trained DistilBERT backbone for SNLI (~85%), we train from scratch (~63-67%)
- 🎨 **Visualizations:** Run `python scripts/visualize_results.py` for comprehensive bar graphs comparing our results with paper's claims


---

## Quick Start

```bash
# Train both models on Wikitext-2
./speedrun.sh all wikitext

# Train on SNLI
./speedrun.sh all snli

# Evaluation only mode (uses latest checkpoints)
./speedrun.sh grassmann wikitext eval

# Train only (skip evaluation)
./speedrun.sh transformer snli train
```

---

## Overview

This repository contains a complete reproduction of the Grassmann flow architecture for language modeling. The original paper claims Grassmann flows achieve perplexity "within 10-15% of size-matched Transformers" on Wikitext-2. **Our reproduction reveals a significantly larger gap of 31-47%** - approximately **3-4x worse than claimed**.

However, we found a **surprising result on SNLI**: Grassmann models **outperform** Transformers by 3.73% when trained from scratch (66.36% vs 62.63%), suggesting geometric operations may be better suited for natural language inference than for language modeling.

**Wikitext-2 Results:**

The paper reports validation perplexities of:
- L=128, N=6: Transformer 248.4 vs Grassmann 275.7 (11% gap)
- L=256, N=12: Transformer 235.2 vs Grassmann 261.1 (11% gap)

Our best validation perplexities are:
- L=128, N=6: Transformer 181.21 vs Grassmann 252.09 (39% gap)
- L=256, N=12: Transformer 169.25 vs Grassmann 241.60 (43% gap)

**SNLI Results:**

Paper (with DistilBERT): Transformer 85.11% vs Grassmann 85.38% (+0.27%)  
Our reproduction: Transformer 62.79% vs Grassmann 66.47% (+3.68%)

### Key Results

#### Wikitext-2 Language Modeling (Best Validation PPL)

| Model       | Config      | Parameters | Best Val PPL | Best Epoch | Gap from Best |
|-------------|-------------|-----------|--------------|------------|---------------|
| **Transformer** | L=256, N=12 | 17.36M    | **169.25**   | 8          | baseline      |
| **Transformer** | L=128, N=12 | 17.32M    | 174.68       | 7          | +3.2%         |
| **Transformer** | L=128, N=6  | 12.59M    | 181.21       | 7          | +7.1%         |
| **Transformer** | L=256, N=6  | 12.62M    | 181.21       | 7          | +7.1%         |
| **Grassmann**   | L=256, N=12 | 17.41M    | 241.60       | 7          | +42.8%        |
| **Grassmann**   | L=128, N=12 | 17.37M    | 242.69       | 6          | +43.4%        |
| **Grassmann**   | L=256, N=6  | 12.64M    | 250.84       | 6          | +48.2%        |
| **Grassmann**   | L=128, N=6  | 12.61M    | 252.09       | 6          | +49.0%        |

**Performance Gap:** Grassmann models show 43-49% higher validation perplexity than comparable Transformers  
**Paper's Claim:** 10-11% gap  
**Our Finding:** 43-49% gap (approximately **4x larger** than claimed)

#### SNLI Natural Language Inference (Test Set)

| Model       | Accuracy | Loss   | Entailment | Neutral | Contradiction | Parameters |
|-------------|----------|--------|------------|---------|---------------|-----------|
| **Grassmann** | **66.36%**   | 0.7624 | 72.68%     | 61.17%  | 64.94%        | 17.70M    |
| **Transformer** | 62.63%   | 0.8398 | 71.44%     | 56.76%  | 59.31%        | 17.67M    |

**Surprising Finding:** Grassmann **outperforms** Transformer on SNLI (+3.68% accuracy) when trained from scratch!

**Paper's Claims (with DistilBERT backbone):**
- Transformer head: 85.11% test accuracy
- Grassmann-Plücker head: 85.38% test accuracy (+0.27%)

**Our Results (from-scratch training):**
- Transformer: 62.79% test accuracy
- Grassmann: 66.47% test accuracy (+3.68%)


---

## Background: What are Grassmann Flows?

### The Core Idea

Instead of using attention to mix token representations, Grassmann flows use **geometric operations on Grassmann manifolds**:

1. **Project** each token representation to a lower dimension (r=32)
2. **Compute Plucker coordinates** between tokens at different positions
3. **Mix** the geometric features with the original representation through learned gating

### Plucker Coordinates

For two vectors `u, v` in R^r, the Plucker coordinates are:

```
p_ij = u_i * v_j - u_j * v_i    for all i < j
```

This produces `r(r-1)/2` coordinates (496 for r=32) that encode the geometric relationship between the vectors. The key property: **antisymmetry** (p_ij = -p_ji), which naturally encodes directional relationships.

### Why Might This Work?

- **Geometric structure**: Captures relationships through manifold geometry instead of dot products
- **Linear complexity**: Avoids attention's O(n²) scaling
- **Theoretical elegance**: Connections to projective geometry and Lie groups

### Why Might This Struggle?

- **Fixed operations**: Unlike attention, the geometric operation is predetermined (not content-dependent)
- **Information bottleneck**: 496 Plucker dimensions compressed back to 256
- **Simple averaging**: All window sizes weighted equally, unlike attention's learned weighting
- **Antisymmetry mismatch**: Language relationships may not be naturally antisymmetric

---

## Architecture Details

### Grassmann Model (Exact Paper Specification)

```python
GrassmannGPT(
    vocab_size=50257,        # GPT2 tokenizer
    max_seq_len=256,
    model_dim=256,
    num_layers=6,
    reduced_dim=32,          # r value for Plucker
    window_sizes=[1,2,4,8,12,16],
    ff_dim=1024,             # 4x model_dim
    dropout=0.1,
    tie_weights=True
)
```

**Layer Structure:**
1. Plucker coordinate computation
2. L2 normalization
3. Linear projection (496 → 256)
4. Average across window sizes
5. Gating: `alpha * h + (1-alpha) * geo_features`
6. LayerNorm
7. Feed-forward network

**Total Parameters:** 17,695,168 (17.70M)

### Transformer Baseline

```python
BaseTransformer(
    vocab_size=50257,
    max_seq_len=256,
    model_dim=256,
    num_layers=6,
    num_heads=8,
    ff_dim=1024,
    dropout=0.1,
    tie_weights=True
)
```

**Total Parameters:** 17,670,400 (17.67M)

---

## Training Configuration

```python
optimizer = AdamW(lr=3e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(T_max=epochs)
batch_size = 32
gradient_clip = 1.0
epochs = 20
```

### Dataset: Wikitext-2

- **Train:** 9,343 chunks (256 tokens each)
- **Validation:** 965 chunks
- **Test:** 1,106 chunks
- **Tokenizer:** GPT2Tokenizer (vocab 50,257)

---

## Reproduction Results

### Best Validation Results vs Paper

**Note:** Training now tracks best validation checkpoints. All evaluation results below use `best.pt` checkpoints.

#### Wikitext-2: Our Results vs Paper's Claims

| Config | Model | Our Params | Our Test PPL | Paper Params | Paper Val PPL | Our Gap | Paper Gap |
|--------|-------|------------|--------------|--------------|---------------|---------|-----------|
| **L128 N6** | Transformer | 12.59M | 189.51 | 12.59M | 248.4 | baseline | baseline |
| **L128 N6** | Grassmann | 12.61M | 261.58 | 13.00M | 275.7 | **+38.0%** | +11.0% |
| **L256 N12** | Transformer | 17.36M | 177.74 | 17.32M | 235.2 | baseline | baseline |
| **L256 N12** | Grassmann | 17.41M | 256.42 | 18.16M | 261.1 | **+44.3%** | +11.0% |

**Key Finding:** Our gap (38-44%) is **3.5-4x larger** than paper's claim (11%)

#### SNLI: Our Results vs Paper's Claims

| Model | Our Test Acc | Paper Test Acc (DistilBERT) | Our Gap | Paper Gap |
|-------|--------------|------------------------------|---------|-----------|
| Transformer | 62.79% | 85.11% | baseline | baseline |
| Grassmann | 66.47% | 85.38% | **+3.68%** | +0.27% |

**Key Finding:** Grassmann **outperforms** Transformer in both our results and paper's (when trained from scratch vs with DistilBERT backbone)

**Checkpoint Information:**
- Training scripts now save `best.pt` checkpoint based on best validation metric
- Evaluations use best checkpoints by default (see `speedrun.sh`)
- Best validation results from latest run (outputs/2026-01-09_19-25-27):
  - Wikitext Transformer L256 N12: Val PPL 169.25 (epoch 8) → Test PPL 177.74
  - Wikitext Grassmann L256 N12: Val PPL 241.60 (epoch 7) → Test PPL 256.42
  - SNLI Transformer: Test Accuracy 62.79%
  - SNLI Grassmann: Test Accuracy 66.47%

### Detailed Evaluation Results

#### Wikitext-2 Test Set Performance

**Transformer Models:**

| Configuration | Layers (N) | Seq Length (L) | Params  | Test Loss | Test PPL | Improvement |
|--------------|-----------|----------------|---------|-----------|----------|-------------|
| Best Config  | 12        | 256            | 17.36M  | 5.1803    | 177.74   | baseline    |
| N12 L128     | 12        | 128            | 17.32M  | 5.2143    | 183.89   | -3.5%       |
| N6 L128      | 6         | 128            | 12.59M  | 5.2445    | 189.51   | -6.6%       |
| N6 L256      | 6         | 256            | 12.62M  | 5.2521    | 190.98   | -7.4%       |

**Grassmann Models:**

| Configuration | Layers (N) | Seq Length (L) | Params  | Test Loss | Test PPL | vs Transformer |
|--------------|-----------|----------------|---------|-----------|----------|----------------|
| N6 L128      | 6         | 128            | 12.61M  | 5.5668    | 261.58   | +38.0%         |
| N6 L256      | 6         | 256            | 12.64M  | 5.5632    | 260.66   | +36.5%         |
| N12 L128     | 12        | 128            | 17.37M  | 5.5500    | 257.23   | +39.9%         |
| N12 L256     | 12        | 256            | 17.41M  | 5.5468    | 256.42   | +44.3%         |

**Key Observations:**
- **Best Transformer:** L=256, N=12 achieves 177.74 perplexity
- **Best Grassmann:** L=256, N=12 achieves 256.42 perplexity
- **Performance Gap:** 36-44% depending on configuration
- **Scaling:** Both models benefit from more layers, with N=12 L=256 performing best
- **Context Length:** Longer sequences (L=256) benefit both architectures

#### SNLI Test Set Performance

**Grassmann Model Results:**

| Split      | Overall Acc | Loss   | Entailment Acc | Neutral Acc | Contradiction Acc | Samples |
|------------|-------------|--------|----------------|-------------|-------------------|---------|
| Test       | **66.47%**  | 0.7628 | 72.89%         | 61.08%      | 65.15%           | 9,824   |

**Transformer Model Results:**

| Split      | Overall Acc | Loss   | Entailment Acc | Neutral Acc | Contradiction Acc | Samples |
|------------|-------------|--------|----------------|-------------|-------------------|---------|
| Test       | **62.79%**  | 0.8372 | 71.38%         | 57.19%      | 59.41%           | 9,824   |

**Comparison:**
- **Grassmann outperforms Transformer by 3.68%** on SNLI (66.47% vs 62.79%)
- Both models struggle most with neutral classification
- Grassmann shows better per-class accuracy across all categories

**Paper's Results (with DistilBERT backbone):**
- Transformer head: 85.11% test
- Grassmann-Plücker head: 85.38% test
- Gap: +0.27% (Grassmann better)

**Our Results (from-scratch):**
- Transformer: 62.79% test
- Grassmann: 66.47% test
- Gap: +3.68% (Grassmann better)

### Visualizations

Generate comprehensive bar graphs:

```bash
python scripts/visualize_results.py
```

This creates:
- `images/wikitext_results.png` - Wikitext-2 perplexity comparison (our test vs paper's validation)
- `images/snli_results.png` - SNLI accuracy comparison (Transformer vs Grassmann, our vs paper)

**Paper's Claims vs Our Results:**
- **Wikitext-2 (Val PPL):** Paper reports 10-11% gap (Transformer 235.2-248.4 vs Grassmann 261.1-275.7)
- **Our Wikitext-2 (Test PPL):** We observe 36-44% gap (Transformer 177.74-190.98 vs Grassmann 256.42-261.58)
- **SNLI:** Paper reports Grassmann wins by 0.27% (85.38% vs 85.11%); our results show Grassmann wins by 3.68% (66.47% vs 62.79%)

---

## Repository Structure

```
attention_is_not_all_you_need/
├── speedrun.sh                 # Main training script
├── scripts/
│   ├── visualize_results.py   # Generate result visualizations
│   └── analyze_best_results.py # Analyze best validation checkpoints
├── images/                     # Generated visualization graphs
│   ├── wikitext_results.png   # Wikitext-2 comparison
│   └── snli_results.png       # SNLI comparison
├── src/
│   └── attn_is_not_all_you_need/
│       ├── models/
│       │   ├── grassmann.py   # Grassmann model implementation
│       │   ├── transformer.py # Baseline transformer
│       │   └── snli_models.py # SNLI classification heads
│       ├── data/
│       │   ├── wikitext.py    # Wikitext-2 dataloader
│       │   └── snli.py        # SNLI dataloader
│       ├── train.py           # Training loop
│       ├── train_snli.py      # SNLI training
│       ├── eval_wikitext.py   # Wikitext evaluation
│       └── eval_snli.py       # SNLI evaluation
├── outputs/                    # Training outputs
│   └── 2026-01-09_13-33-09/   # Latest training run
│       ├── grassmann_*/       # Grassmann model checkpoints
│       ├── transformer_*/     # Transformer model checkpoints
│       └── analysis/          # Performance analysis
├── blog.md                     # Detailed analysis
└── technical.md               # Technical report
```

---

## Usage Examples

### Basic Training

```bash
# Train Grassmann on Wikitext-2 (L=128 and L=256)
./speedrun.sh grassmann wikitext

# Train Transformer on SNLI
./speedrun.sh transformer snli

# Train both models on all datasets
./speedrun.sh all all
```

### Configuration Options

```bash
# Override layer depths (default: 6, 12)
LAYER_DEPTHS_OVERRIDE=6 ./speedrun.sh all wikitext

# Both 6 and 12 layer models
LAYER_DEPTHS_OVERRIDE=6,12 ./speedrun.sh all wikitext
```

### Evaluation Modes

```bash
# Eval only - uses most recent checkpoints
./speedrun.sh grassmann wikitext eval

# Train only - skip evaluation
./speedrun.sh all snli train

# Default - both train and eval
./speedrun.sh transformer wikitext
```


### Analysis

Analyze and compare best validation results:

```bash
# Extract best validation checkpoints and compare with paper
python scripts/analyze_best_results.py

# Outputs:
#   - Markdown table comparing our results vs paper's claims
#   - outputs/*/best_results_summary.json - Comprehensive JSON summary
```

**Note:** Training scripts automatically save `best.pt` checkpoints based on best validation metrics. Evaluations use these by default.

---

## Analysis of Discrepancy

### Paper's Claims vs Our Findings

#### Wikitext-2 Language Modeling

**Paper's Reported Results (Validation PPL):**

| Model | Layers | Params | Val PPL | Gap |
|-------|--------|--------|---------|-----|
| TransformerLM (L=128) | 6 | 12.59M | 248.4 | baseline |
| GrassmannLM (L=128) | 6 | 13.00M | 275.7 | +11.0% |
| TransformerLM (L=256) | 12 | 17.32M | 235.2 | baseline |
| GrassmannLM (L=256) | 12 | 18.16M | 261.1 | +11.0% |

**Our Reproduction Results (Test PPL):**

| Model | Layers | Params | Test PPL | Gap |
|-------|--------|--------|----------|-----|
| TransformerLM (L=128) | 6 | 12.59M | 236.35 | baseline |
| GrassmannLM (L=128) | 6 | 12.61M | 310.13 | +31.2% |
| TransformerLM (L=256) | 12 | 17.36M | 220.21 | baseline |
| GrassmannLM (L=256) | 12 | 17.41M | 316.00 | +43.5% |

**Key Discrepancy:** Our gap (31-47%) is **3-4x larger** than the paper's claim (10-15%)

#### SNLI Classification

**Paper's Reported Results (with DistilBERT backbone):**

| Model | Val Accuracy | Test Accuracy |
|-------|-------------|---------------|
| Transformer head | 85.45% | 85.11% |
| Grassmann-Plücker head | 85.50% | 85.38% |

Gap: +0.27% (Grassmann slightly better)

**Our Results (from-scratch training):**

| Model | Test Accuracy | Entailment | Neutral | Contradiction |
|-------|---------------|------------|---------|---------------|
| Transformer | 62.63% | 71.44% | 56.76% | 59.31% |
| Grassmann | 66.36% | 72.68% | 61.17% | 64.94% |

Gap: +3.73% (Grassmann significantly better)

**Key Insight:** Grassmann outperforms Transformer on SNLI in both paper's results and ours, suggesting the geometric operations may be better suited for NLI tasks compared to language modeling.

### Possible Explanations for Gap

The paper leaves several details unspecified:

**Hyperparameters:**
- Learning rate warmup schedule
- Weight initialization schemes
- Per-layer window size variations
- Exact dropout placement

**Implementation Details:**
- Normalization order variations
- Gating initialization strategies
- Residual connection formulations
- Exact activation functions

**Experimental Setup:**
- Tokenization preprocessing
- Chunk boundary handling
- Evaluation protocol specifics
- Random seed sensitivity

### What We Verified

✅ Exact architecture from paper (v4 implementation)  
✅ Specified hyperparameters (lr, dropout, dimensions)  
✅ Dataset and preprocessing  
✅ Training procedure  
✅ Size-matched comparison (17.7M params)  

