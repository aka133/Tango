import torch
import numpy as np
import time

def benchmark_torch(M, N, K):
    a = torch.randn(M, K, device='cuda')
    b = torch.randn(K, N, device='cuda')

    # Torch
    try:
        start = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"torch: {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"torch failed: {e}")

    # Numpy
    try:
        a_np = a.cpu().numpy()
        b_np = b.cpu().numpy()
        start = time.perf_counter()
        np.matmul(a_np, b_np)
        elapsed = time.perf_counter() - start
        print(f"numpy: {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"numpy failed: {e}")

if __name__ == "__main__":
    benchmark_torch(256, 256, 256)