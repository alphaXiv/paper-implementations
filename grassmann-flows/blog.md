# Grassmann Flows for Sequence Modeling: An Independent Reproduction Study

**Author:** Elliot Arledge
**Date:** December 2025
**Hardware:** NVIDIA H100 SXM5 80GB (Voltage Park Cloud)

---

## Abstract

I present an independent reproduction study of "Attention Is Not What You Need" (arXiv 2512.19428), which proposes replacing transformer self-attention with Grassmann manifold-based geometric operations using Plucker coordinates. The original paper claims Grassmann flow layers achieve perplexity "within 10-15% of size-matched Transformers" on Wikitext-2. My reproduction, using the exact architecture specified in the paper, reveals a **22.6% performance gap** - significantly larger than claimed. This report details the reproduction methodology, analyzes potential sources of discrepancy, situates Grassmann flows within the broader landscape of attention alternatives (Mamba, RWKV, Linear Attention), and offers theoretical perspectives on why geometric approaches may face fundamental challenges in language modeling.

---

## 1. Introduction

The transformer architecture has dominated sequence modeling since 2017, with self-attention providing a powerful mechanism for capturing long-range dependencies. However, the quadratic complexity of attention with respect to sequence length has motivated an extensive search for alternatives. Recent years have seen the emergence of state space models (Mamba), linear recurrent units (RWKV), and various forms of linear attention - all attempting to match transformer quality while achieving better computational scaling.

Into this landscape comes a provocative proposal: replacing attention entirely with operations on Grassmann manifolds. The paper "Attention Is Not What You Need" argues that the geometric structure of Grassmann manifolds - specifically through Plucker coordinate embeddings - can capture the pairwise token interactions that attention provides, while maintaining linear complexity in sequence length.

This is an appealing idea. Grassmann manifolds have deep connections to projective geometry and have found applications in computer vision and subspace tracking. The paper's claim of achieving performance "within 10-15% of size-matched Transformers" would make Grassmann flows a competitive alternative worthy of serious consideration.

I set out to reproduce these results exactly. What I found complicates the narrative.

---

## 2. Background: Grassmann Manifolds and Plucker Coordinates

### 2.1 What is a Grassmann Manifold?

The Grassmann manifold Gr(k, n) is the space of all k-dimensional linear subspaces of an n-dimensional vector space. Unlike Euclidean space, the Grassmann manifold has non-trivial curvature - it is a smooth, compact manifold where "points" are themselves subspaces rather than vectors.

For the simplest non-trivial case, Gr(1, n) is the projective space of lines through the origin - a single vector (up to scaling) defines each point. Gr(2, n) is the space of planes through the origin. These geometric objects have natural notions of distance, angles, and interpolation that differ fundamentally from Euclidean geometry.

### 2.2 Plucker Coordinates: A Concrete Embedding

Given two vectors u, v in R^r that span a 2-dimensional subspace (a plane through the origin), the Plucker coordinates provide a concrete way to represent this subspace. For vectors with components u = (u_1, ..., u_r) and v = (v_1, ..., v_r), the Plucker coordinates are:

```
p_ij = u_i * v_j - u_j * v_i    for all i < j
```

This produces r(r-1)/2 coordinates - for r=32 (the paper's value), that's 496 Plucker coordinates per token pair.

The key property: Plucker coordinates are **antisymmetric** (p_ij = -p_ji) and satisfy the **Plucker relations**, a set of quadratic constraints that characterize exactly which vectors correspond to valid 2-planes. They encode the "wedge product" or exterior algebra structure - the signed area/volume relationships between vector components.

### 2.3 Intuition: Why Might This Work for Sequences?

The paper's core insight is this: in a sequence of tokens, each token at position t has a hidden representation h_t. By projecting this to a lower dimension z_t and computing Plucker coordinates between z_t and z_{t-delta} (for various window sizes delta), we capture **geometric relationships** between token representations at different positions.

Where attention asks "how much should token t attend to token s?" via a dot-product similarity, Plucker coordinates ask "what is the geometric relationship between the subspaces defined by these token representations?" The antisymmetric structure means forward and backward relationships are explicitly different - causality is baked into the geometry.

This is elegant. Whether it is *effective* is the empirical question.

---

## 3. The Original Paper's Claims

The paper "Attention Is Not What You Need" (arXiv 2512.19428) makes several specific claims:

1. **Performance**: Grassmann flow layers achieve perplexity "within 10-15% of size-matched Transformers" on Wikitext-2
2. **Model Size**: Experiments use 13-18M parameter models
3. **Architecture Specifics**:
   - Reduced dimension r = 32
   - Window sizes: {1, 2, 4, 8, 12, 16} for 6-layer models
   - Blend gating: alpha * h + (1-alpha) * g
   - Gate input: concatenation of [h; g]
   - L2 normalization of Plucker coordinates before projection
   - Layer order: Plucker -> L2 norm -> Projection -> Average -> Gate -> LayerNorm

The paper positions Grassmann flows as a geometrically-motivated alternative to attention, noting theoretical connections to Lie groups and differential geometry.

---

## 4. Related Work: The Landscape of Attention Alternatives

Before presenting my reproduction results, it is worth situating Grassmann flows within the broader context of attention alternatives.

### 4.1 Mamba and State Space Models

Mamba represents the state-space-model (SSM) approach to linear-complexity sequence modeling. Key characteristics:

- **Selective state spaces**: Input-dependent state transitions (unlike earlier SSMs like S4)
- **Hardware efficiency**: Custom CUDA kernels achieve 5x higher inference throughput than transformers
- **Scaling**: Mamba-3B outperforms transformers of the same size; matches transformers twice its size

NVIDIA's 2024 study comparing 8B-parameter models found that pure Mamba matches or exceeds transformers on many tasks, but lags on copying/in-context learning. Their Mamba-2-Hybrid (43% Mamba-2, 7% attention, 50% MLP) exceeds pure transformers on all 12 benchmarks by +2.65 points average.

### 4.2 RWKV: Linear RNNs at Scale

RWKV achieves transformer-competitive performance with true O(n) complexity through clever recurrent formulations:

- **Scale**: Successfully trained to 14B parameters - the largest dense RNN ever
- **Efficiency**: 10-100x lower inference costs compared to transformers
- **Real-world deployment**: RWKV v5 (Eagle) ships on 1.5 billion Windows machines for Copilot
- **Benchmarks**: Eagle 7B achieves Lambada perplexity of 3.36 (vs Mistral 7B's 3.18)

Like Mamba, RWKV struggles with associative recall - a 70M attention model outperforms a 1.4B gated-convolution model on this task.

### 4.3 Linear Attention and Hybrids

Linear attention approximates softmax attention through kernel methods, achieving O(nd^2) instead of O(n^2 d):

- **BASED**: Strongest sub-quadratic architecture on recall, +6.22 accuracy over Mamba, but still below transformers
- **Kimi Linear**: 75% KV cache reduction, up to 6x decoding throughput
- **The hybrid trend**: Modern approaches mix 3-6:1 linear-to-full attention ratios

The consensus: pure linear attention struggles with recall, but hybridization works.

### 4.4 Where Does Grassmann Fit?

Grassmann flows represent a distinct approach: rather than approximating attention or using recurrent state, they propose entirely different operations based on geometric manifold structure. This makes them theoretically interesting but empirically unproven - hence the importance of reproduction.

---

## 5. Methodology: Exact Reproduction

### 5.1 Implementation

I implemented GrassmannGPT to match the paper's architecture exactly:

```python
class CausalGrassmannMixing(nn.Module):
    """
    Paper's forward pass:
    1. z_t = W_red * h_t + b_red
    2. For each delta: p_ij = z_t_i * z_{t-delta}_j - z_t_j * z_{t-delta}_i
    3. p_hat = p / max(||p||_2, eps)  [L2 normalize]
    4. g_t^(delta) = W_plu * p_hat + b_plu  [project each window]
    5. g_t = average(g_t^(delta)) across valid deltas
    6. alpha = sigmoid(W_gate * [h_t; g_t] + b_gate)  [gate from concat]
    7. h_mix = alpha * h_t + (1-alpha) * g_t  [blend, not add]
    8. Apply LayerNorm
    """
```

Key implementation details:
- **Plucker computation**: Vectorized using index buffers for efficiency
- **L2 normalization**: Applied per-token with eps=1e-8 for stability
- **Gating**: Blend formula (not additive) with sigmoid from concatenated inputs
- **Window sizes**: Exactly [1, 2, 4, 8, 12, 16] as specified

### 5.2 Model Configurations

**GrassmannGPT:**
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

**Size-Matched Transformer Baseline:**
```
Parameters: 17,670,400 (17.67M)
- model_dim: 256
- num_layers: 6
- num_heads: 8
- ff_dim: 1024 (4x model_dim)
- dropout: 0.1
- tie_weights: True
```

Both models are within 0.14% of each other in parameter count - a fair comparison.

### 5.3 Dataset: Wikitext-2

Following the paper, I used Wikitext-2 (wikitext-2-raw-v1):
- **Tokenizer**: GPT2Tokenizer (vocab size: 50,257)
- **Sequence length**: 256 tokens
- **Train**: 9,343 chunks
- **Validation**: 965 chunks
- **Test**: 1,106 chunks

### 5.4 Training Configuration

```
Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
Scheduler: CosineAnnealingLR
Epochs: 20
Batch size: 32
Gradient clipping: 1.0
Hardware: NVIDIA H100 SXM5 80GB
Training time: ~30 minutes per model
Throughput: ~53k tokens/sec (Grassmann), similar for Transformer
```

---

## 6. Results

### 6.1 Training Curves

**Grassmann Model:**

| Epoch | Train Loss | Val Loss | Val PPL |
|-------|------------|----------|---------|
| 1     | 7.2473     | 6.4292   | 619.66  |
| 5     | 5.2780     | 5.6617   | 287.65  |
| 10    | 4.7262     | 5.4898   | 242.21  |
| 15    | 4.5029     | 5.4640   | 236.05  |
| 20    | 4.4482     | 5.4654   | 236.36  |

**Transformer Model:**

| Epoch | Train Loss | Val Loss | Val PPL |
|-------|------------|----------|---------|
| 1     | 7.1584     | 6.3960   | 599.42  |
| 5     | 5.2462     | 5.5120   | 247.64  |
| 10    | 4.5300     | 5.2681   | 194.04  |
| 13    | 4.3043     | 5.2472   | 190.03  |
| 20    | 4.1350     | 5.2630   | 193.05  |

### 6.2 Final Comparison

| Model       | Parameters | Best Val PPL | Test PPL |
|-------------|------------|--------------|----------|
| Grassmann   | 17.70M     | 236.05       | 242.94   |
| Transformer | 17.67M     | 190.03       | 198.17   |

### 6.3 Gap Analysis

```
Test PPL Ratio: 242.94 / 198.17 = 1.226
Observed Gap: 22.6%
Paper Claim: 10-15%
Discrepancy: +7.6% to +12.6% worse than claimed
```

**The paper's claim is not reproduced.** The observed 22.6% gap significantly exceeds the claimed 10-15%.

---

## 7. Analysis

### 7.1 What the Numbers Mean

A 22.6% perplexity gap is substantial. To put this in context:

- Perplexity roughly measures how "surprised" the model is by the test data
- A perplexity of 198 means the model's average uncertainty is equivalent to choosing among ~198 equally likely next tokens
- A perplexity of 243 means ~243 equally likely tokens
- The gap represents meaningfully worse language modeling

### 7.2 Convergence Behavior

Looking at the training curves:

1. **Early training**: Both models start similarly (epoch 1 PPL within 3%)
2. **Mid-training divergence**: By epoch 5, the transformer pulls ahead
3. **Late training**: Grassmann shows diminishing returns after epoch 15; transformer continues improving
4. **Overfitting**: Both show slight validation degradation in final epochs

The transformer achieves its best validation PPL at epoch 13 (190.03), while Grassmann peaks at epoch 15 (236.05) - both begin overfitting thereafter, but at very different performance levels.

### 7.3 Potential Sources of Discrepancy

The paper leaves several details unspecified:

**Hyperparameters:**
- Learning rate schedule warmup steps and minimum LR
- Weight initialization scheme (I used Xavier for projections, 0.02 std for embeddings)
- Per-layer window size variations (paper mentions this possibility)
- Dropout placement specifics

**Implementation Ambiguities:**
- Pre-norm vs post-norm variations beyond what's stated
- Sigmoid initialization (I used zeros for gate weights, zeros for bias)
- Exact residual connection formulation
- FFN activation (I used GELU as is standard)

**Experimental Setup:**
- Tokenization preprocessing details
- Chunk boundary handling (I used non-overlapping chunks)
- Evaluation protocol (sliding window vs chunk-based)
- Random seed sensitivity

### 7.4 Earlier Experiments: OpenWebText

Before this exact reproduction, I ran experiments on OpenWebText with larger models:

| Model | Parameters | Val PPL | Notes |
|-------|------------|---------|-------|
| GPT-2 | 85M | 223 | Standard baseline |
| GrassmannGPTv3 | 48M | 312 | Modified: r=64, additive gating |
| GrassmannGPT | 43M | 350 | Paper exact: r=32, blend gating |

These were not size-matched comparisons on the paper's dataset, but the trend is consistent: the paper's exact architecture (v4) actually performed *worse* than my modified version (v3), and both significantly underperformed GPT-2.

---

## 8. Theoretical Discussion

### 8.1 Why Might Plucker Coordinates Struggle?

Setting aside implementation details, there are theoretical reasons to question whether Grassmann geometry is well-suited for language modeling:

**1. Fixed Geometric Operations**

Attention computes input-dependent weights: the relevance of token s to token t depends on the *content* of both tokens at runtime. Plucker coordinates, by contrast, perform a fixed geometric operation (the antisymmetric wedge product) on projected representations. The learning happens in the projections, but the core mixing operation is predetermined.

This is fundamentally different from Mamba's selective state spaces, which learn input-dependent state transitions, or from attention, which learns content-based routing.

**2. The r(r-1)/2 Bottleneck**

With r=32, the Plucker embedding has 496 dimensions. This is projected back to model_dim=256 - meaning substantial information compression. Compare to attention, where the full model_dim participates in key-query matching.

Increasing r rapidly increases Plucker dimensions (r=64 gives 2016 dims), creating a computational/memory tradeoff. The paper's choice of r=32 may be suboptimal, but larger values were not explored.

**3. Window Averaging vs. Learned Aggregation**

The paper averages Plucker features across window sizes. This treats delta=1 (adjacent tokens) and delta=16 (distant tokens) equally. Attention, by contrast, learns position-dependent biases (through positional embeddings) and content-dependent weights (through softmax).

Simple averaging may not capture the complex, context-dependent weighting that language requires.

**4. Antisymmetry May Not Match Language Structure**

Plucker coordinates satisfy p_ij = -p_ji - the relationship of component i to j is the negative of j to i. While this enforces causality (forward differs from backward), language relationships are not generally antisymmetric.

The phrase "the cat sat" involves "cat" relating to "sat" (subject-verb), but the reverse relationship is not simply the negation. Linguistic roles are asymmetric in more complex ways than antisymmetry captures.

### 8.2 Comparison to Successful Alternatives

The architectures that successfully challenge attention share common properties:

**Mamba/SSMs:**
- Input-dependent state transitions (selectivity)
- Custom hardware optimization
- Successful hybridization with attention

**RWKV:**
- Learned time-decay mechanisms
- Channel-wise mixing learned from data
- Attention-like mechanisms in later versions (RWKV v6)

**Linear Attention:**
- Kernel approximations that preserve attention's fundamental structure
- Hybrid approaches that keep some softmax attention

The successful alternatives generally preserve some form of input-dependent, learned aggregation. They approximate or modify attention rather than replacing it with fundamentally different operations.

Grassmann flows take a different path: fixed geometric operations with learned projections. The geometric elegance may not translate to the flexible, content-dependent mixing that language modeling requires.

### 8.3 The Recall Problem

All sub-quadratic alternatives struggle with **associative recall** - the ability to remember and retrieve specific information from earlier in the context. This is where attention excels: explicitly storing and retrieving key-value pairs.

Grassmann flows have no explicit memory mechanism. Information about past tokens is compressed into the Plucker coordinates and averaged across windows. Long-range dependencies must be captured through the geometry, but the fixed operations and averaging may lose the specificity needed for recall.

The BASED architecture's success on recall comes from explicitly addressing this with specialized mechanisms. Grassmann flows, as presented, have no comparable solution.

---

## 9. Alternative Implementation: My v3 Modifications

During development, I created GrassmannGPTv3 with modifications intended to improve stability and performance:

```python
# v3 differences from paper (v4):
- reduced_dim = 64 (vs 32)
- window_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
- Additive gating: output = hidden + gate * geo_features
- Skip L2 normalization, use LayerNorm after projection
- Gate initialized to favor residual (bias=-2.0)
- Smaller weight initialization (std=0.01)
```

Despite these attempts at stabilization, v3 still underperformed v4 on the paper's Wikitext-2 setup, and both underperformed transformers. This suggests the issues are more fundamental than hyperparameter tuning.

---

## 10. Conclusions

### 10.1 Summary of Findings

1. **Paper claims not reproduced**: My 22.6% gap significantly exceeds the claimed 10-15%
2. **Consistent underperformance**: Across multiple configurations and datasets, Grassmann flows underperform transformers
3. **Theoretical concerns**: Fixed geometric operations may not provide the flexible, content-dependent mixing that language requires
4. **Missing details**: The paper's specifications leave room for implementation variation, but I followed all stated details exactly

### 10.2 Is Grassmann Dead?

Not necessarily. Several paths forward exist:

1. **Hybridization**: Like Mamba-2-Hybrid, Grassmann layers might work better combined with some attention
2. **Learned aggregation**: Replace averaging with learned attention-like weights over windows
3. **Input-dependent geometry**: Make the geometric operations selective based on input, like Mamba's selectivity
4. **Different tasks**: Grassmann geometry might suit domains with more inherent geometric structure (3D, vision)

### 10.3 The Broader Lesson

The search for attention alternatives remains active and important. Mamba, RWKV, and linear attention variants demonstrate that competitive performance is achievable with sub-quadratic complexity. But the successful approaches share a common thread: they preserve or approximate the content-dependent, learned aggregation that makes attention powerful.

Grassmann flows represent a genuinely different approach - replacing attention with geometric operations. This is intellectually interesting and worth exploring. But my reproduction suggests the current formulation falls short of competitive performance, and the gap exceeds what the paper claims.

Independent reproduction is essential. When a paper claims results that challenge established baselines, the community benefits from verification. In this case, the verification reveals a more nuanced picture: Grassmann flows are a creative idea that, in their current form, do not deliver on the stated performance claims.

---

## 11. CUDA Kernel Optimization

As part of this reproduction, I implemented custom CUDA kernels for the Plucker coordinate computation - the computational bottleneck of Grassmann flows.

### 11.1 Implementation

The CUDA kernel fuses the following operations:
1. Plucker coordinate computation across all window sizes
2. L2 normalization
3. Averaging across valid windows

This avoids multiple kernel launches and intermediate memory allocations in the naive PyTorch implementation.

### 11.2 Performance Results

Benchmarked on NVIDIA H100 80GB HBM3 with a 17.7M parameter model:

| Metric | PyTorch | CUDA | Speedup |
|--------|---------|------|---------|
| Mixing layer forward | 1.59 ms | 0.35 ms | **4.6x** |
| Full model inference | 9.16 ms | 4.53 ms | **2.0x** |
| Inference throughput | 0.45M tok/s | 0.90M tok/s | **2.0x** |

### 11.3 Correctness Verification

I verified the CUDA implementation produces identical results to PyTorch:
- 100 iterations of training with both implementations
- Loss differences: 0.0001-0.0002 (numerical precision level)
- Identical convergence behavior

### 11.4 Limitations

The backward pass currently uses a Python fallback with per-dimension loops, limiting training speedup. The forward-only optimization is most beneficial for inference scenarios.

---

## 12. Reproducibility and Code

All code and results are available:

```bash
# Reproduce on any machine with GPU
python train_wikitext2.py --model both --epochs 20 --model-dim 256 --num-layers 6
```

**Key files:**
- `train_wikitext2.py` - Training script
- `src/models/grassmann.py` - Exact paper implementation
- `src/cuda/grassmann_kernels.cu` - CUDA kernel implementation
- `src/cuda/grassmann_fused.py` - PyTorch wrapper for CUDA kernels
- `test_cuda_kernels.py` - CUDA correctness and benchmark tests

**Hardware used:** NVIDIA H100 SXM5 80GB (Voltage Park Cloud)

**Results JSON:**
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

---

## References and Sources

### Primary Paper
- "Attention Is Not What You Need" (arXiv 2512.19428)

### Attention Alternatives
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Mamba-360 Survey](https://arxiv.org/html/2404.16112v1)
- [NVIDIA Empirical Study of Mamba-based Language Models](https://research.nvidia.com/publication/2024-06_empirical-study-mamba-based-language-models)
- [The Evolution of RWKV](https://arxiv.org/html/2411.02795)
- [RWKV at COLM 2024](https://arxiv.org/pdf/2404.19178)
- [Eagle 7B Analysis](https://medium.com/ai-insights-cobet/eagle-7-billion-how-the-rwkv-model-surpasses-traditional-transformer-based-models-in-ai-71dbc98ce383)
- [Gated Linear Attention Transformers](https://arxiv.org/pdf/2312.06635)
- [BASED: Linear Attention Recall-Throughput Tradeoff](https://www.together.ai/blog/based)
- [2024 in Post-Transformers Architectures](https://www.latent.space/p/2024-post-transformers)
- [Zoology: Measuring Recall in Efficient Language Models](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)

### Grassmann Geometry
- Absil, P-A., et al. "Optimization Algorithms on Matrix Manifolds" (Princeton University Press, 2008)
- Edelman, A., et al. "The Geometry of Algorithms with Orthogonality Constraints" (SIAM Journal, 1998)

---

*This reproduction study was conducted as an independent investigation. The author has no affiliation with the original paper's authors. All experiments were run on cloud GPU infrastructure (Voltage Park) with total compute cost under $10.*

---

## Citation

If you find this reproduction study useful, please cite:

```bibtex
@article{arledge2025grassmann,
  title={Grassmann Flows for Sequence Modeling: An Independent Reproduction Study},
  author={Arledge, Elliot},
  year={2025},
  month={December},
  url={https://github.com/Infatoshi/grassmann-flows}
}
```

Or in prose:

> Arledge, E. (2025). "Grassmann Flows for Sequence Modeling: An Independent Reproduction Study." GitHub: https://github.com/Infatoshi/grassmann-flows
