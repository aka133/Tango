import torch
import ctypes
import time
import numpy as np
import os

# Set environment variables
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load CUDA libraries
cuda_lib = ctypes.CDLL('./libcuda_kernels.so')

# Define argument types
cuda_lib.launch_float4_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
cuda_lib.launch_double_buffer_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

def pad_to_multiple(size, multiple=4):
    return ((size + multiple - 1) // multiple) * multiple

def benchmark_cuda(M, N, K):
    # Pad dimensions to multiples of 4
    M_padded = pad_to_multiple(M)
    N_padded = pad_to_multiple(N)
    K_padded = pad_to_multiple(K)

    # Ensure aligned memory allocation with padded dimensions
    a = torch.zeros((M_padded, K_padded), device='cuda', dtype=torch.float32)
    b = torch.zeros((K_padded, N_padded), device='cuda', dtype=torch.float32)
    c = torch.zeros((M_padded, N_padded), device='cuda', dtype=torch.float32)

    # Ensure memory is contiguous
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()

    # Fill with random values after allocation
    a.normal_()
    b.normal_()

    # Float4 Kernel
    try:
        start = time.perf_counter()
        cuda_lib.launch_float4_matmul(a.data_ptr(), b.data_ptr(), c.data_ptr(), M_padded, N_padded, K_padded)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"float4: {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"float4 failed: {e}")

    # Double Buffer Kernel
    try:
        start = time.perf_counter()
        cuda_lib.launch_double_buffer_matmul(a.data_ptr(), b.data_ptr(), c.data_ptr(), M_padded, N_padded, K_padded)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"double_buffer: {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"double_buffer failed: {e}")

if __name__ == "__main__":
    # Use dimensions that are multiples of 4
    benchmark_cuda(256, 256, 256)