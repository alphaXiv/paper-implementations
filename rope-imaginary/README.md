<h1>Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs</h1>

## Introduction

This repo provides two implementations of [Rope++](https://www.alphaxiv.org/abs/2512.07525) an imaginary extension of Rotary Position Embeddings. 

The basis of the Rope++ work is that when doing standard RoPE, we are throwing away the imaginary component of the attention scores when computing QK^T. RoPE treats a vector [a,b] as a complex number z = a + bi in order to cleanly apply a rotation via multiplication by e^(iθ). However, in the computation of QK^T, we are implicitly only taking the real part of the complex dot product. The paper finds that having some attention heads be the standard real part of the complext dot product and the rest of the heads be the imaginary part perform better over long-contexts. 

In this implementation we provide both a cleaned-up adaptation of the official "Rope++" codebase as well as an implementation of "Rope++" with Karpathy's Nanochat repo. For the cleaned-up version of the original Rope++ codebase, we greatly simplify the setup, training, and evaluation process, providing a Nanochat-style 'speedrun' bash script which sets up UV, installs the necessary packages, and runs the training+evaluation scripts all in one go.