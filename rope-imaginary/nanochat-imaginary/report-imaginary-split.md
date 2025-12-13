# nanochat training report

Generated: 2025-12-12 01:34:45

## Environment

### Git Information
- Branch: master
- Commit: 679edf5 (dirty)
- Message: config saving

### Hardware
- Platform: Linux
- CPUs: 104 cores (208 logical)
- Memory: 1771.7 GB
- GPUs: 8x NVIDIA H100 80GB HBM3
- GPU Memory: 633.5 GB total
- CUDA Version: 12.8
- Hourly Rate: $24.00/hour

### Software
- Python: 3.10.12
- PyTorch: 2.8.0+cu128


### Bloat
- Characters: 430,905
- Lines: 10,784
- Files: 51
- Tokens (approx): 107,726
- Dependencies (uv.lock lines): 2,218

Run started: 2025-12-12 01:34:47

---

## Tokenizer training
timestamp: 2025-12-12 01:36:16

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 65,536
- train_time: 82.9977
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.9151
- token_bytes_std: 2.8736


## Tokenizer evaluation
timestamp: 2025-12-12 01:36:20

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 375 | 4.85 | +7.2% |
| korean | 893 | 745 | 1.20 | 721 | 1.24 | +3.2% |
| code | 1259 | 576 | 2.19 | 493 | 2.55 | +14.4% |
| math | 1834 | 936 | 1.96 | 966 | 1.90 | -3.2% |
| science | 1112 | 260 | 4.28 | 225 | 4.94 | +13.5% |
| fwe-train | 4208518 | 900364 | 4.67 | 856901 | 4.91 | +4.8% |
| fwe-val | 4908443 | 1059062 | 4.63 | 1010356 | 4.86 | +4.6% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 375 | 4.85 | +3.1% |
| korean | 893 | 364 | 2.45 | 721 | 1.24 | -98.1% |
| code | 1259 | 309 | 4.07 | 493 | 2.55 | -59.5% |
| math | 1834 | 832 | 2.20 | 966 | 1.90 | -16.1% |
| science | 1112 | 249 | 4.47 | 225 | 4.94 | +9.6% |
| fwe-train | 4208518 | 874799 | 4.81 | 856901 | 4.91 | +2.0% |
| fwe-val | 4908443 | 1029691 | 4.77 | 1010356 | 4.86 | +1.9% |


## Base model training
timestamp: 2025-12-12 04:38:30

- run: d20-imaginary-split
- device_type: 
- depth: 20
- max_seq_len: 2048
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 20
- device_batch_size: 32
- total_batch_size: 524,288
- embedding_lr: 0.2000
- unembedding_lr: 0.0040
- weight_decay: 0.0000
- matrix_lr: 0.0200
- grad_clip: 1.0000
- warmup_ratio: 0.0000
- warmdown_ratio: 0.2000
- final_lr_frac: 0.0000
- resume_from_step: -1
- eval_every: 250
- eval_tokens: 10,485,760
- core_metric_every: 2000
- core_metric_max_per_task: 500
- sample_every: 2000
- save_every: -1
- use_imaginary: True
- model_tag: 
- Number of parameters: 560,988,160
- Number of FLOPs per token: 3.491758e+09
- Calculated number of iterations: 21,400
- Number of training tokens: 11,219,763,200
- Tokens : Params ratio: 20.0000
- DDP world size: 8
- warmup_ratio: 0.0000
- warmdown_ratio: 0.2000
- final_lr_frac: 0.0000
- Minimum validation bpb: 0.8120
- Final validation bpb: 0.8120
- CORE metric estimate: 0.2119
- MFU %: 48.20%
- Total training flops: 3.917670e+19
- Total training time: 171.90m
- Peak memory usage: 75422.52MiB


## Base model loss
timestamp: 2025-12-12 04:39:08

- train bpb: 0.8148
- val bpb: 0.8121
- sample 0: <|bos|>The capital of France is Paris. The capital of France is Paris. The capital of France is Paris.
- sample 1: <|bos|>The chemical symbol of gold is Au. The chemical symbol of gold is Au. The atomic number of gold is
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Monday. If today is Tuesday, then tomorrow will be Wednesday. If today is
- sample 3: <|bos|>The opposite of hot is cold. The opposite of cold is hot. The opposite of hot is cold.
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune,
- sample 5: <|bos|>My favorite color is red. I like to wear red. I like to wear red. I like
- sample 6: <|bos|>If 5*x + 3 = 13, then x is 3/5. If x is 3/5, then x is


## Base model evaluation
timestamp: 2025-12-12 04:42:50

- Model: base_model (step 21400)
- CORE metric: 0.2096
- hellaswag_zeroshot: 0.2562
- jeopardy: 0.1049
- bigbench_qa_wikidata: 0.5332
- arc_easy: 0.5202
- arc_challenge: 0.1172
- copa: 0.3600
- commonsense_qa: 0.0715
- piqa: 0.3711
- openbook_qa: 0.1307
- lambada_openai: 0.3641
- hellaswag: 0.2635
- winograd: 0.2381
- winogrande: 0.0687
- bigbench_dyck_languages: 0.0630
- agi_eval_lsat_ar: 0.0761
- bigbench_cs_algorithms: 0.3909
- bigbench_operators: 0.1714
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.2297
- coqa: 0.2061
- boolq: -0.1049
- bigbench_language_identification: 0.1792


## Midtraining
timestamp: 2025-12-12 04:51:30

- run: d20-imaginary-split
- device_type: 
- dtype: bfloat16
- num_iterations: -1
- max_seq_len: 2048
- device_batch_size: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- init_lr_frac: 1.0000
- weight_decay: 0.0000
- eval_every: 150
- eval_tokens: 10,485,760
- total_batch_size: 524,288
- dry_run: 0
- Number of iterations: 809
- DDP world size: 8
- Minimum validation bpb: 0.3950


## Chat evaluation mid
timestamp: 2025-12-12 04:58:47

- source: mid
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.4440
- ARC-Challenge: 0.3003
- MMLU: 0.3312
- GSM8K: 0.0379
- HumanEval: 0.0915
- SpellingBee: 0.9844
- ChatCORE metric: 0.2580


## Chat SFT
timestamp: 2025-12-12 05:01:14

- run: d20-imaginary-split
- source: mid
- device_type: 
- dtype: bfloat16
- device_batch_size: 4
- num_epochs: 1
- num_iterations: -1
- target_examples_per_step: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0200
- eval_every: 100
- eval_steps: 100
- eval_metrics_every: 200
- eval_metrics_max_problems: 1024
- Training rows: 22,439
- Number of iterations: 701
- Training loss: 0.5320
- Validation loss: 1.0100


## Chat evaluation sft
timestamp: 2025-12-12 05:07:13

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.4415
- ARC-Challenge: 0.3072
- MMLU: 0.3267
- GSM8K: 0.0682
- HumanEval: 0.0915
- SpellingBee: 0.9922
- ChatCORE metric: 0.2643


## Summary

- Characters: 430,905
- Lines: 10,784
- Files: 51
- Tokens (approx): 107,726
- Dependencies (uv.lock lines): 2,218

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2096   | -        | -        | -        |
| ARC-Challenge   | -        | 0.3003   | 0.3072   | -        |
| ARC-Easy        | -        | 0.4440   | 0.4415   | -        |
| GSM8K           | -        | 0.0379   | 0.0682   | -        |
| HumanEval       | -        | 0.0915   | 0.0915   | -        |
| MMLU            | -        | 0.3312   | 0.3267   | -        |
| ChatCORE        | -        | 0.2580   | 0.2643   | -        |

Total wall clock time: 3h32m
