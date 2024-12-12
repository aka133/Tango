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

## CUDA Development Setup

### Cloud Environment Setup (e.g., Lambda Labs, Vast.ai)

1. **Configure CUDA Environment**
```bash
# Set Triton's ptxas path
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
echo 'export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas' >> ~/.bashrc
source ~/.bashrc
```

2. **Install Miniconda**
```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make installer executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Run installer (accept license & defaults)
./Miniconda3-latest-Linux-x86_64.sh

# Source conda initialization script directly
source ~/miniconda3/etc/profile.d/conda.sh

# Initialize conda in your current shell
conda init

# Activate conda changes
source ~/.bashrc

# Verify installation
conda --version
```

3. **Create and Activate Conda Environment**
```bash
conda create -n tango python=3.10
conda activate tango
```

4. **Install Required Packages**
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install CuPy for CUDA operations
conda install -c conda-forge cupy cuda-version=12.1

# Install Triton for GPU kernel development
conda install -c conda-forge triton
```

5. **Set Python Path**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/workspace/Tango"

# For persistence, add to ~/.bashrc:
echo 'export PYTHONPATH="${PYTHONPATH}:/workspace/Tango"' >> ~/.bashrc
source ~/.bashrc
```

### Project Structure
```
Tango/
├── centigrad/
│   ├── __init__.py
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── triton_kernels.py
│   │   ├── cuda_kernels.cu
│   │   └── cudnn_utils.cu
```

### Verify Installation
```python
import torch
import triton
import cupy

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Triton version: {triton.__version__}")
print(f"CuPy version: {cupy.__version__}")
```

### Common Issues

1. **PYTHONPATH not set correctly**
   - Symptom: ModuleNotFoundError when importing project modules
   - Solution: Ensure PYTHONPATH includes the project root directory

2. **CUDA version mismatch**
   - Symptom: Runtime errors related to CUDA
   - Solution: Ensure all CUDA-related packages (PyTorch, CuPy, Triton) use compatible versions

3. **Missing NVIDIA drivers**
   - Symptom: CUDA not available
   - Solution: Install appropriate NVIDIA drivers for your GPU

### Development Tools

- **NVIDIA Nsight Systems**: For system-wide performance analysis
- **NVIDIA Nsight Compute**: For detailed kernel analysis
- **Triton Kernel Debugger**: For debugging custom Triton kernels

### References

- [PyTorch Documentation](https://pytorch.org/docs)
- [Triton Documentation](https://triton-lang.org/main/getting-started/installation.html)
- [CuPy Documentation](https://docs.cupy.dev/en/stable/install.html)