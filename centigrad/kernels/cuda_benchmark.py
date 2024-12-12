import torch
import ctypes
import time
from cuda_kernels import float4_coalesced_matmul, double_buffering_loop_unrolling_matmul

def benchmark_cuda(M, N, K):
    a = torch.randn(M, K, device='cuda')
    b = torch.randn(K, N, device='cuda')
    c = torch.zeros(M, N, device='cuda')

    # Float4 Kernel
    try:
        start = time.perf_counter()
        float4_coalesced_matmul(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"float4: {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"float4 failed: {e}")

    # Double Buffer Kernel
    try:
        start = time.perf_counter()
        double_buffering_loop_unrolling_matmul(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"double_buffer: {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"double_buffer failed: {e}")

if __name__ == "__main__":
    benchmark_cuda(256, 256, 256)