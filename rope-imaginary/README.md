<h1>Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs</h1>

## Introduction

This repo provides two implementations of [Rope++](https://www.alphaxiv.org/abs/2512.07525) an imaginary extension of Rotary Position Embeddings. 

For a refresher, the basis of RoPE is to apply varying rotations across the query and key vectors (as opposed to adding an absolute position vector like in the Attention paper). These rotations are done by segmenting the input vector into vectors of dimension 2 and applying a rotational transformation e^(iθ). These complex terms are discarded when computing attention scores.

The basis of Rope++ is that the imaginary component of the attention score contains important information and should be considered in half of the attention heads. The imaginary attention component uses a sine-based characteristic curve which decays much more slowly over distance compared to the cosine-based curve in standard RoPE's real attention. This slower decay means imaginary attention maintains stronger weights for distant tokens rather than emphasizing only nearby tokens, allowing it to better capture long-range dependencies in the sequence.

In this implementation we provide both a cleaned-up adaptation of the official "Rope++" codebase as well as an implementation of "Rope++" with Karpathy's Nanochat repo. For the cleaned-up version of the original Rope++ codebase, we greatly simplify the setup, training, and evaluation process, providing a Nanochat-style 'speedrun' bash script which sets up UV, installs the necessary packages, and runs the training+evaluation scripts all in one go.