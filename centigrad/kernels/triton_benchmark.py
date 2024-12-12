import torch
import time
from triton_kernels import matmul  # Your Triton implementation

def benchmark_triton(M, N, K):
    a = torch.randn(M, K, device='cuda')
    b = torch.randn(K, N, device='cuda')

    try:
        start = time.perf_counter()
        matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"triton: {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"triton failed: {e}")

if __name__ == "__main__":
    benchmark_triton(256, 256, 256)