import torch
import ctypes
import time
import numpy as np
import os

# Set environment variables
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load CUDA libraries
cuda_lib = ctypes.CDLL('./libcuda_kernels.so')

# Define argument types
cuda_lib.launch_float4_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
cuda_lib.launch_double_buffer_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

def benchmark_cuda(M, N, K):
    a = torch.randn(M, K, device='cuda')
    b = torch.randn(K, N, device='cuda')
    c = torch.zeros(M, N, device='cuda')

    # Float4 Kernel
    try:
        start = time.perf_counter()
        cuda_lib.launch_float4_matmul(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"float4: {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"float4 failed: {e}")

    # Double Buffer Kernel
    try:
        start = time.perf_counter()
        cuda_lib.launch_double_buffer_matmul(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"double_buffer: {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"double_buffer failed: {e}")

if __name__ == "__main__":
    benchmark_cuda(256, 256, 256)