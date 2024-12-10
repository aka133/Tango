# Tango

This project is an active exercise in building an LLM and associated evaluation/red-teaming from scratch, from the autograd and tokenization to GPU kernels and deployment. 

I started by creating my own autograd library, using Andrej Karpathy's micrograd library for most of the foundation and adding in support for SwiGLU, tensor operations and operations like batchnorm.

I've also added custom CUDA kernels for matmuls, optimizing with shared memory tiling, float4 vectorization and double buffering. 

The next steps are:
- Manually written Triton matmul kernels and evaluating performance compared to optimization with CuBLAS and custom CUTLASS kernels. Will compare performance using NSight.
- From-scratch implementation of byte-pair encoding for tokenization (using support from HuggingFace until then)
- Multi-GPU support
- MCTS integration for o1-like reasoning capabilities
- Open-source AI evals
- Designing a redteaming framework
- Interpretability features (inspired by [Li et al.'s recent work]([url](https://arxiv.org/abs/2410.19750)) on geometric approaches to interpretability)
- Deployment on a home Kubernetes cluster

Overall, this project is intended as a constant learning experience and features are subject to change.
