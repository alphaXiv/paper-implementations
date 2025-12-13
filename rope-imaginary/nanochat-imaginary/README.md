<h1>Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs</h1>

## Introduction

This is a fork of Karpathy's [Nanochat](https://github.com/karpathy/nanochat) repo. We modify the apply_rotary_emb step, introducing an 'apply_rotary_emb_imaginary' function that can run on either half of the original heads, or doubling the total num_heads count. We run Rope++ in the 'split' configuration and arrive at similar numbers to the original Nanochat. That being said, Rope++ excels in long-context tasks, so we intend on doing further long-context tuning and evaluation shortly with Nanochat + Rope++.