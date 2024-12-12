import time
import numpy as np
import torch
import triton
import cupy as cp
import ctypes
from ctypes import c_void_p, c_int, c_float, POINTER
from centigrad.kernels import (
    matmul
)
from centigrad import Value
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx
import os

# Set CUDA DSA before any CUDA operations
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load CUDA libraries
_libcublas = ctypes.CDLL('libcublas.so')
# _libcudnn = ctypes.CDLL('libcudnn.so')

# Define handle types
cublasHandle_t = c_void_p
# cudnnHandle_t = c_void_p

# Create/Destroy handle functions
_libcublas.cublasCreate_v2.restype = c_int
_libcublas.cublasCreate_v2.argtypes = [POINTER(cublasHandle_t)]
_libcublas.cublasDestroy_v2.restype = c_int
_libcublas.cublasDestroy_v2.argtypes = [cublasHandle_t]

# Load our compiled CUDA libraries
cuda_lib = ctypes.CDLL('./libcuda_kernels.so')
cublas_lib = ctypes.CDLL('./libcublas_utils.so')
# cudnn_lib = ctypes.CDLL('./libcudnn_utils.so')

cuda_lib.float4_coalesced_matmul.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
cuda_lib.double_buffering_loop_unrolling_matmul.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
cublas_lib.cublas_matmul.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_float, c_float]

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.init()
torch.cuda.set_device(0)

class KernelBenchmark:
    def __init__(self,
                 matmul_sizes=[(1024, 1024, 1024)],
                 ln_sizes=[(512, 768)],
                 tolerance=1e-3,
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
        
    #    handle = cudnnHandle_t()
    #    status = _libcudnn.cudnnCreate(ctypes.byref(handle))
    #    if status != 0:
    #        raise RuntimeError(f"cuDNN initialization failed with status {status}")
    #    self.cudnn_handle = handle
    
    def __del__(self):
        # Cleanup handles
        if hasattr(self, 'cublas_handle'):
            _libcublas.cublasDestroy_v2(self.cublas_handle)

    def verify_result(self, result, reference, name):
        """Verify kernel output matches reference implementation"""
        if result is None:
            print(f"ERROR: {name} returned None!")
            return False
        
        # Convert everything to numpy for comparison
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        elif isinstance(result, cp.ndarray):
            result = result.get()
        
        # Print more debug info
        print(f"\nDebug {name}:")
        print(f"Result shape: {result.shape}")
        print(f"Result range: [{result.min():.6f}, {result.max():.6f}]")
        print(f"Result mean: {result.mean():.6f}")
        print(f"Result std: {result.std():.6f}")
        
        if isinstance(reference, torch.Tensor):
            reference = reference.cpu().numpy()
        elif isinstance(reference, cp.ndarray):
            reference = reference.get()
        
        # Compute relative error instead of absolute
        max_diff = np.max(np.abs(result - reference))
        rel_diff = np.max(np.abs((result - reference) / (reference + 1e-7)))
        
        is_close = np.allclose(result, reference, rtol=self.tolerance, atol=self.tolerance)

        if not is_close:
            print(f"WARNING: {name} results differ from reference")
            print(f"Max absolute difference: {max_diff}")
            print(f"Max relative difference: {rel_diff}")
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
            
            try:
                # Create inputs using uniform instead of randn for large matrices
                if M * K > 16 * 1024 * 1024:  # If larger than 16M elements
                    print("Large matrix detected, using uniform distribution...")
                    a_torch = torch.empty(M, K, device='cuda').uniform_(-1, 1)
                    b_torch = torch.empty(K, N, device='cuda').uniform_(-1, 1)
                else:
                    a_torch = torch.randn(M, K, device='cuda')
                    b_torch = torch.randn(K, N, device='cuda')
                
                c_torch = torch.zeros((M, N), device='cuda')
                
                # Simple timing for each implementation
                implementations = {
                    "numpy": lambda: np.matmul(a_torch.cpu().numpy(), b_torch.cpu().numpy()),
                    "torch": lambda: torch.matmul(a_torch, b_torch),
                    "triton": lambda: matmul(a_torch, b_torch),
                    "cublas": lambda: (
                        print(f"\nCUBLAS params: M={M}, N={N}, K={K}"),
                        cublas_lib.cublas_matmul(
                            ctypes.c_void_p(int(a_torch.data_ptr())),
                            ctypes.c_void_p(int(b_torch.data_ptr())),
                            ctypes.c_void_p(int(c_torch.data_ptr())),
                            ctypes.c_int(M),
                            ctypes.c_int(N),
                            ctypes.c_int(K),
                            ctypes.c_float(1.0),
                            ctypes.c_float(0.0)
                        ),
                        c_torch
                    )[1],
                    "float4": lambda: (
                        cuda_lib.launch_float4_matmul(
                            ctypes.c_void_p(int(a_torch.data_ptr())),
                            ctypes.c_void_p(int(b_torch.data_ptr())),
                            ctypes.c_void_p(int(c_torch.data_ptr())),
                            M, N, K
                        ),
                        torch.cuda.synchronize(),
                        c_torch
                    )[2],
                    "double_buffer": lambda: (
                        cuda_lib.launch_double_buffer_matmul(
                            ctypes.c_void_p(int(a_torch.data_ptr())),
                            ctypes.c_void_p(int(b_torch.data_ptr())),
                            ctypes.c_void_p(int(c_torch.data_ptr())),
                            M, N, K
                        ),
                        torch.cuda.synchronize(),
                        c_torch
                    )[2]
                }
                
                results = {}
                for name, impl in implementations.items():
                    try:
                        # Warmup
                        for _ in range(3):
                            _ = impl()
                            torch.cuda.synchronize()
                        
                        # Time it
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(10):
                            _ = impl()
                            torch.cuda.synchronize()
                        end = time.perf_counter()
                        
                        avg_time = (end - start) / 10
                        tflops = (2 * M * N * K) / (avg_time * 1e12)
                        print(f"{name}: {avg_time*1000:.2f}ms ({tflops:.1f} TFLOPS)")
                        
                        results[name] = {"time": avg_time, "tflops": tflops}
                        
                    except Exception as e:
                        print(f"{name} failed: {str(e)}")
                
                self.results[f"{M}x{N}x{K}"] = results
                
            except Exception as e:
                print(f"Failed to benchmark size {M}x{N}x{K}: {str(e)}")
                continue

    def debug_small_matmul(self):
        """Debug with a small matrix multiplication case"""
        M, N, K = 4, 4, 4
        
        # Create known input matrices
        a_np = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float32)
        
        b_np = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float32)
        
        # Expected result
        reference = np.matmul(a_np, b_np)
        print("Reference result:")
        print(reference)
        
        # Test PyTorch
        a_torch = torch.from_numpy(a_np).cuda()
        b_torch = torch.from_numpy(b_np).cuda()
        c_torch = torch.zeros((M, N), dtype=torch.float32, device="cuda")
        
        # Test CUBLAS
        c_torch.zero_()
        cublas_lib.cublas_matmul(
            ctypes.c_void_p(int(a_torch.data_ptr())),
            ctypes.c_void_p(int(b_torch.data_ptr())),
            ctypes.c_void_p(int(c_torch.data_ptr())),
            M, N, K,
            ctypes.c_float(1.0),
            ctypes.c_float(0.0)
        )
        torch.cuda.synchronize()
        print("\nCUBLAS result:")
        print(c_torch.cpu().numpy())

    '''
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
    '''
    def print_results(self):
        """Pretty print results with comparison to baseline."""
        for size in self.matmul_sizes:
            M, N, K = size
            size_str = f"{M}x{N}x{K}"
            print(f"\nMatMul Results for size {size_str}:")
            print(f"{'Implementation':<15} {'Time (ms)':<10} {'TFLOPS':<10}")
            print("-" * 35)
            
            if size_str in self.results:
                for name, stats in self.results[size_str].items():
                    print(f"{name:<15} {stats['time']*1000:>9.2f} {stats['tflops']:>9.1f}")

def main():
    # Keep smaller sizes for now
    matmul_sizes = [
        (256, 256, 256),       # Small square case
        (512, 512, 512),       # Medium square case
        (1024, 1024, 1024),    # Large square case
    ]
    
    ln_sizes = [
        (512, 768),     # Base size
        (128, 768),     # Small batch
        (2048, 768),    # Large batch
    ]
    
    # Force CUDA initialization before any operations
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.cuda.synchronize()  # Make sure GPU is ready
    
    benchmark = KernelBenchmark(matmul_sizes, ln_sizes)
    benchmark.benchmark_matmul()
    benchmark.print_results()

if __name__ == "__main__":
    main()