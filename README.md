# Tango

This project is an active exercise in building an LLM and associated evaluation/red-teaming from scratch, from the autograd and tokenization to GPU kernels and deployment. 

I started by creating my own autograd library, using Andrej Karpathy's micrograd library for most of the foundation and adding in support for SwiGLU, tensor operations and operations like batchnorm.

I've added:
- Custom CUDA kernels for matmuls, optimizing with shared memory tiling, float4 vectorization and double buffering
- Manually written Triton matmul kernels
- cuBLAS matmul wrapper
- cuDNN layernorm wrapper
- Performance evaluation for all matmul/layernorm acceleration using NSight

The next steps are:
- From-scratch implementation of byte-pair encoding for tokenization
- Multi-GPU support
- MCTS integration for o1-like reasoning capabilities (using Alibaba's open-source Marco-o1)
- Open-source AI evals (EleutherAI, MMLU)
- Designing a redteaming framework
- Interpretability features (inspired by [Li et al.'s recent work]([url](https://arxiv.org/abs/2410.19750)) on geometric approaches to interpretability)
- Deployment on a home Kubernetes cluster

Overall, this project is intended as a constant learning experience and features are subject to change.
