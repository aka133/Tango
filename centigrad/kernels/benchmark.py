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

            # Initialize inputs with controlled range
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            a_np = rng.uniform(-1, 1, (M, K)).astype(np.float32)
            b_np = rng.uniform(-1, 1, (K, N)).astype(np.float32)
            reference = np.matmul(a_np, b_np)

            # GPU tensors
            a_torch = torch.from_numpy(a_np).cuda()
            b_torch = torch.from_numpy(b_np).cuda()
            c_torch = torch.zeros((M, N), dtype=torch.float32, device="cuda")

            # Debug: Print input statistics
            print(f"Input A - min: {a_np.min():.6f}, max: {a_np.max():.6f}")
            print(f"Input B - min: {b_np.min():.6f}, max: {b_np.max():.6f}")
            print(f"Reference - min: {reference.min():.6f}, max: {reference.max():.6f}")

            # Verify PyTorch result first as sanity check
            torch_result = torch.matmul(a_torch, b_torch)
            torch_np = torch_result.cpu().numpy()
            print("\nPyTorch matmul check:")
            print(f"PyTorch result - min: {torch_np.min():.6f}, max: {torch_np.max():.6f}")
            print(f"Difference stats - min: {(torch_np - reference).min():.6f}, max: {(torch_np - reference).max():.6f}")

            # Test CUBLAS specifically
            c_torch.zero_()  # Ensure clean output tensor
            cublas_lib.cublas_matmul(
                ctypes.c_void_p(int(a_torch.data_ptr())),
                ctypes.c_void_p(int(b_torch.data_ptr())),
                ctypes.c_void_p(int(c_torch.data_ptr())),
                M, N, K,
                ctypes.c_float(1.0),  # alpha
                ctypes.c_float(0.0)   # beta
            )
            torch.cuda.synchronize()
            cublas_result = c_torch.cpu().numpy()
            print("\nCUBLAS check:")
            print(f"CUBLAS result - min: {cublas_result.min():.6f}, max: {cublas_result.max():.6f}")
            print(f"Difference stats - min: {(cublas_result - reference).min():.6f}, max: {(cublas_result - reference).max():.6f}")

            # GPU tensors
            a_torch = torch.from_numpy(a_np).cuda()
            b_torch = torch.from_numpy(b_np).cuda()
            c_torch = torch.zeros((M, N), dtype=torch.float32, device="cuda")

            # Ensure GPU is synchronized before benchmarks
            torch.cuda.synchronize()

            # Benchmark implementations
            print("\nDebug Triton inputs:")
            print(f"Input shapes: A={a_torch.shape}, B={b_torch.shape}, C={c_torch.shape}")
            print(f"Input devices: A={a_torch.device}, B={b_torch.device}, C={c_torch.device}")
            print(f"Input strides: A={a_torch.stride()}, B={b_torch.stride()}, C={c_torch.stride()}")
            print(f"Input dtypes: A={a_torch.dtype}, B={b_torch.dtype}, C={c_torch.dtype}")
            print(f"Dimensions: M={M}, N={N}, K={K}")

            implementations = {
                "numpy_cpu": (lambda a, b: np.matmul(a, b), (a_np, b_np)),
                "torch": (lambda a, b: torch.matmul(a, b), (a_torch, b_torch)),
                "cublas": (
                    lambda a, b, c: (
                        cublas_lib.cublas_matmul(
                            ctypes.c_void_p(int(a.data_ptr())),
                            ctypes.c_void_p(int(b.data_ptr())),
                            ctypes.c_void_p(int(c.data_ptr())),
                            M, N, K,
                            ctypes.c_float(1.0),
                            ctypes.c_float(0.0)
                        ),
                        c  # Return the output tensor
                    )[1],
                    (a_torch, b_torch, c_torch)
                ),
                "float4_coalesced": (
                    lambda a, b, c: (
                        cuda_lib.float4_coalesced_matmul(
                            ctypes.c_void_p(int(a.data_ptr())),
                            ctypes.c_void_p(int(b.data_ptr())),
                            ctypes.c_void_p(int(c.data_ptr())),
                            M, N, K
                        ),
                        torch.cuda.synchronize(),  # Make sure kernel is done
                        c  # Return the output tensor
                    )[2],  # Get the tensor from the tuple
                    (a_torch, b_torch, c_torch)
                ),
                "double_buffering": (
                    lambda a, b, c: (
                        cuda_lib.double_buffering_loop_unrolling_matmul(
                            ctypes.c_void_p(int(a.data_ptr())),
                            ctypes.c_void_p(int(b.data_ptr())),
                            ctypes.c_void_p(int(c.data_ptr())),
                            M, N, K
                        ),
                        torch.cuda.synchronize(),  # Make sure kernel is done
                        c  # Return the output tensor
                    )[2],  # Get the tensor from the tuple
                    (a_torch, b_torch, c_torch)
                ),
                "triton": (
                    lambda a, b, c: (
                        print("\nCalling Triton kernel..."),
                        c.copy_(matmul(a, b).float()),
                        print("Triton kernel completed"),
                        c  # Return c as the result
                    )[1],
                    (a_torch, b_torch, c_torch)
                )
            }

            for name, (fn, inputs) in implementations.items():
                result = self.benchmark_fn(fn, f"matmul_{name}", inputs, reference)
                if result:
                    # Add TFLOPS calculation for matmul
                    result["tflops"] = (2 * M * N * K) / (result["mean"] * 1e-12)
                    self.results[f"{name}_{M}x{N}x{K}"] = result

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
        
        '''
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
        '''

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
    benchmark.debug_small_matmul()  # Add this line

    # Run benchmarks with Nsight profiling enabled
    with torch.cuda.profiler.profile():
        benchmark.benchmark_matmul()
        # benchmark.benchmark_layernorm()
    
    benchmark.print_results()

if __name__ == "__main__":
    main()