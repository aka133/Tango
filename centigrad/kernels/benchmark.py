import time
import numpy as np
import torch
import triton
import cupy as cp
import ctypes
from ctypes import c_void_p, c_int, c_float, POINTER
from centigrad.kernels import (
    matmul1,
    matmul2,
)
from centigrad import Value
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx

# Load CUDA libraries
_libcublas = ctypes.CDLL('libcublas.so')
_libcudnn = ctypes.CDLL('libcudnn.so')

# Define handle types
cublasHandle_t = c_void_p
cudnnHandle_t = c_void_p

# Create/Destroy handle functions
_libcublas.cublasCreate_v2.restype = c_int
_libcublas.cublasCreate_v2.argtypes = [POINTER(cublasHandle_t)]
_libcublas.cublasDestroy_v2.restype = c_int
_libcublas.cublasDestroy_v2.argtypes = [cublasHandle_t]

# Load our compiled CUDA libraries
cuda_lib = ctypes.CDLL('./libcuda_kernels.so')
cublas_lib = ctypes.CDLL('./libcublas_utils.so')
cudnn_lib = ctypes.CDLL('./libcudnn_utils.so')

cuda_lib.float4_coalesced_matmul.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
cuda_lib.double_buffering_loop_unrolling_matmul.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
cublas_lib.cublas_matmul.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_float, c_float]

class KernelBenchmark:
    def __init__(self,
                 matmul_sizes=[(1024, 1024, 1024)],
                 ln_sizes=[(512, 768)],
                 tolerance=1e-5,
                 warmup=25,
                 rep=100):
        self.matmul_sizes = matmul_sizes
        self.ln_sizes = ln_sizes
        self.tolerance = tolerance
        self.warmup = warmup
        self.rep = rep
        self.results = {}

        # Create handles
        handle = cublasHandle_t()
        status = _libcublas.cublasCreate_v2(ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"cuBLAS initialization failed with status {status}")
        self.cublas_handle = handle
        
        handle = cudnnHandle_t()
        status = _libcudnn.cudnnCreate(ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"cuDNN initialization failed with status {status}")
        self.cudnn_handle = handle
    
    def __del__(self):
        # Cleanup handles
        if hasattr(self, 'cublas_handle'):
            _libcublas.cublasDestroy_v2(self.cublas_handle)

    def verify_result(self, result, reference, name):
        """Verify kernel output matches reference implementation"""
        # Convert everything to numpy for comparison
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        elif isinstance(result, cp.ndarray):
            result = result.get()
        
        if isinstance(reference, torch.Tensor):
            reference = reference.cpu().numpy()
        elif isinstance(reference, cp.ndarray):
            reference = reference.get()
        
        max_diff = np.max(np.abs(result - reference))
        is_close = np.allclose(result, reference, rtol=self.tolerance, atol=self.tolerance)

        if not is_close:
            print(f"WARNING: {name} results differ from reference by {max_diff}")
            return False
        return True
    
    def benchmark_fn(self, fn, name, inputs, reference=None):
        """Generic benchmark function with NSight profiling"""
        # Verify correctness first if reference provided
        if reference is not None:
            result = fn(*inputs)
            if not self.verify_result(result, reference, name):
                print(f"Skipping {name} benchmark due to verification failure")
                return None
            
        # Warmup
        for _ in range(self.warmup):
            _ = fn(*inputs)
            torch.cuda.synchronize()

        # Start profiler
        profiler.start()
        nvtx.range_push(f"{name}_benchmark")

        # Benchmark
        times = []
        for i in range(self.rep):
            nvtx.range_push(f"{name}_iter_{i}")
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = fn(*inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
            nvtx.range_pop()
        
        nvtx.range_pop()
        profiler.stop()

        # Calculate statistics
        times = np.array(times)
        return {
            "mean": np.mean(times),
            "std": np.std(times),
            "median": np.median(times),
            "min": np.min(times),
            "max": np.max(times),
            "verified": reference is not None
        }

    def benchmark_matmul(self):
        """Benchmark all matmul implementations"""
        for M, N, K in self.matmul_sizes:
            print(f"\nBenchmarking matmul with size {M}x{N}x{K}")

            # Initialize inputs
            a_np = np.random.randn(M, K).astype(np.float32)
            b_np = np.random.randn(K, N).astype(np.float32)
            reference = np.matmul(a_np, b_np)

            # GPU tensors
            a_torch = torch.from_numpy(a_np).cuda()
            b_torch = torch.from_numpy(b_np).cuda()
            c_torch = torch.zeros((M, N), dtype=torch.float32, device="cuda")

            # Benchmark implementations
            implementations = {
                "numpy_cpu": (lambda a, b: np.matmul(a, b), (a_np, b_np)),
                "torch": (lambda a, b: torch.matmul(a, b), (a_torch, b_torch)),
                "cublas": (
                    lambda a, b, c: cublas_lib.cublas_matmul(
                        ctypes.c_void_p(int(a.data_ptr())),
                        ctypes.c_void_p(int(b.data_ptr())),
                        ctypes.c_void_p(int(c.data_ptr())),
                        M, N, K,
                        ctypes.c_float(1.0),
                        ctypes.c_float(0.0)
                    ),
                    (a_torch, b_torch, c_torch)
                ),
                "float4_coalesced": (
                    lambda a, b, c: cuda_lib.float4_coalesced_matmul(
                        ctypes.c_void_p(int(a.data_ptr())),
                        ctypes.c_void_p(int(b.data_ptr())),
                        ctypes.c_void_p(int(c.data_ptr())),
                        M, N, K
                    ),
                    (a_torch, b_torch, c_torch)
                ),
                "double_buffering": (
                    lambda a, b, c: cuda_lib.double_buffering_loop_unrolling_matmul(
                        ctypes.c_void_p(int(a.data_ptr())),
                        ctypes.c_void_p(int(b.data_ptr())),
                        ctypes.c_void_p(int(c.data_ptr())),
                        M, N, K
                    ),
                    (a_torch, b_torch, c_torch)
                ),
                "triton_1": (lambda a, b, c: matmul1(a, b, c, M, N, K), (a_torch, b_torch, c_torch)),
                "triton_2": (lambda a, b, c: matmul2(a, b, c, M, N, K), (a_torch, b_torch, c_torch))
            }

            for name, (fn, inputs) in implementations.items():
                result = self.benchmark_fn(fn, f"matmul_{name}", inputs, reference)
                if result:
                    # Add TFLOPS calculation for matmul
                    result["tflops"] = (2 * M * N * K) / (result["mean"] * 1e-12)
                    self.results[f"{name}_{M}x{N}x{K}"] = result

    def benchmark_layernorm(self):
        """Benchmark all layernorm implementations"""
        for batch_size, hidden_dim in self.ln_sizes:
            print(f"\nBenchmarking layernorm with size {batch_size}x{hidden_dim}")

            # Initialize inputs
            x = torch.randn(batch_size, hidden_dim).cuda()
            gamma = torch.ones(hidden_dim, device='cuda')
            beta = torch.zeros(hidden_dim, device='cuda')

            # Reference implementation
            reference = torch.nn.functional.layer_norm(
                x, x.shape[-1:], weight=gamma, bias=beta
            )

            # Benchmark implementations
            implementations = {
                "torch": (reference, (x, )),
                "cudnn": (
                    lambda x: cudnn_lib.cudnn_layernorm(
                        ctypes.c_void_p(int(x.data_ptr())),
                        ctypes.c_void_p(int(gamma.data_ptr())),
                        ctypes.c_void_p(int(beta.data_ptr())),
                        batch_size,
                        hidden_dim,
                        ctypes.c_float(1e-5)
                    ),
                    (x,)
                ),
                "centigrad": (lambda x: Value.layer_norm(x, gamma, beta), (x,))
            }

            for name, (fn, inputs) in implementations.items():
                result = self.benchmark_fn(fn, f"layernorm_{name}", inputs, reference)
                if result:
                    self.results[f"{name}_ln_{batch_size}x{hidden_dim}"] = result

    def print_results(self):
        """Pretty print results with comparison to baseline."""
        # Group results by operation and size
        matmul_results = {k: v for k, v in self.results.items() if k.startswith("matmul")}
        ln_results = {k: v for k, v in self.results.items() if k.startswith("layernorm")}
        
        # Print matmul results
        for size in self.matmul_sizes:
            M, N, K = size
            size_str = f"{M}x{N}x{K}"
            print(f"\nMatMul Results for size {size_str}:")
            print(f"{'Implementation':<15} {'Time (ms)':<10} {'TFLOPS':<10} {'Verified':<10}")
            print("-" * 45)
            
            baseline = None
            for name, stats in matmul_results.items():
                if size_str in name:
                    impl_name = name.split("_")[1]  # Skip "matmul_" prefix
                    if impl_name == "numpy":
                        baseline = stats["mean"]
                    
                    speedup = f"{baseline/stats['mean']:>9.1f}x" if baseline else "baseline"
                    print(f"{impl_name:<15} "
                          f"{stats['mean']*1000:>9.2f} "
                          f"{stats['tflops']:>9.1f} "
                          f"{str(stats['verified']):>9}")
        
        # Print layernorm results
        for size in self.ln_sizes:
            batch_size, hidden_dim = size
            size_str = f"{batch_size}x{hidden_dim}"
            print(f"\nLayerNorm Results for size {size_str}:")
            print(f"{'Implementation':<15} {'Time (ms)':<10} {'Verified':<10}")
            print("-" * 35)
            
            baseline = None
            for name, stats in ln_results.items():
                if size_str in name:
                    impl_name = name.split("_")[1]  # Skip "layernorm_" prefix
                    if impl_name == "torch":
                        baseline = stats["mean"]
                    
                    speedup = f"{baseline/stats['mean']:>9.1f}x" if baseline else "baseline"
                    print(f"{impl_name:<15} "
                          f"{stats['mean']*1000:>9.2f} "
                          f"{str(stats['verified']):>9}")

def main():
    # Example sizes
    matmul_sizes = [
        (1024, 1024, 1024),    # Standard square
        (4096, 4096, 4096),    # Large square
        (8192, 128, 1024),     # Typical attention shape
    ]
    
    ln_sizes = [
        (32, 768),     # Small batch
        (128, 768),    # Medium batch
        (512, 768),    # Large batch
    ]
    
    benchmark = KernelBenchmark(matmul_sizes, ln_sizes)
    
    # Run benchmarks with Nsight profiling enabled
    with torch.cuda.profiler.profile():
        benchmark.benchmark_matmul()
        benchmark.benchmark_layernorm()
    
    benchmark.print_results()

if __name__ == "__main__":
    main()