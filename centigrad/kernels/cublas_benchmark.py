import torch
import ctypes
import time
from cublas_utils import cublas_matmul

def benchmark_cublas(M, N, K):
    a = torch.randn(M, K, device='cuda')
    b = torch.randn(K, N, device='cuda')
    c = torch.zeros(M, N, device='cuda')

    try:
        start = time.perf_counter()
        cublas_matmul(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"cublas: {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"cublas failed: {e}")

if __name__ == "__main__":
    benchmark_cublas(256, 256, 256)