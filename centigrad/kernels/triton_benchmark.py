import torch
import ctypes
import numpy as np
import time
from ctypes import c_void_p, c_int, c_float
from centigrad.kernels import matmul  # Ensure correct import
import os

# Set environment variables
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load CUDA libraries
cuda_lib = ctypes.CDLL('./libcuda_kernels.so')
cublas_lib = ctypes.CDLL('./libcublas_utils.so')

# Define argument types
cuda_lib.launch_float4_matmul.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
cuda_lib.launch_double_buffer_matmul.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
cublas_lib.cublas_matmul.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_float, c_float]

def benchmark_cuda(M, N, K):
    a = torch.randn(M, K, device='cuda')
    b = torch.randn(K, N, device='cuda')
    c = torch.zeros(M, N, device='cuda')

    # Reference result
    reference = torch.matmul(a, b)

    # Float4 Kernel
    try:
        cuda_lib.launch_float4_matmul(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K)
        torch.cuda.synchronize()
        float4_result = c.cpu().numpy()
        max_diff = np.max(np.abs(float4_result - reference.cpu().numpy()))
        print(f"float4: {c.cpu().numpy().mean():.2f}ms | Max Diff: {max_diff:.6f}")
    except Exception as e:
        print(f"float4 failed: {e}")

    # Double Buffer Kernel
    try:
        cuda_lib.launch_double_buffer_matmul(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K)
        torch.cuda.synchronize()
        double_buffer_result = c.cpu().numpy()
        max_diff = np.max(np.abs(double_buffer_result - reference.cpu().numpy()))
        print(f"double_buffer: {c.cpu().numpy().mean():.2f}ms | Max Diff: {max_diff:.6f}")
    except Exception as e:
        print(f"double_buffer failed: {e}")

if __name__ == "__main__":
    benchmark_cuda(256, 256, 256)