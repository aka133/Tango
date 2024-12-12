import torch
import ctypes
import time
import os

# Set environment variables
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load CUBLAS library
cublas_lib = ctypes.CDLL('./libcublas_utils.so')

# Define argument types
cublas_lib.cublas_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]

def benchmark_cublas(M, N, K):
    a = torch.randn(M, K, device='cuda')
    b = torch.randn(K, N, device='cuda')
    c = torch.zeros(M, N, device='cuda')

    try:
        start = time.perf_counter()
        cublas_lib.cublas_matmul(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K, ctypes.c_float(1.0), ctypes.c_float(0.0))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"cublas: {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"cublas failed: {e}")

if __name__ == "__main__":
    benchmark_cublas(256, 256, 256)