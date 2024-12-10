import time
import numpy as np
import torch
import triton # type: ignore
import cupy as cp # type: ignore

class KernelBenchmark:
    def __init__(self, sizes=[(1024, 1024, 1024)], tolerance=1e-5):
        self.sizes = sizes  # List of (M, N, K) for matmul
        self.tolerance = tolerance  # For numerical comparisons
        self.results = {}
        
    def verify_result(self, result, reference, name):
        """Verify kernel output matches reference (numpy) implementation."""
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

    def benchmark_fn(self, fn, name, warmup=25, rep=100):
        """Benchmark a specific implementation."""
        for size in self.sizes:
            M, N, K = size
            print(f"\nBenchmarking {name} with size {M}x{N}x{K}")
            
            # Initialize inputs
            a_np = np.random.randn(M, K).astype(np.float32)
            b_np = np.random.randn(K, N).astype(np.float32)
            
            # Compute reference result
            reference = np.matmul(a_np, b_np)
            
            # Convert inputs to appropriate format
            if name.startswith("cuda"):
                a = cp.array(a_np)
                b = cp.array(b_np)
            elif name.startswith("torch"):
                a = torch.from_numpy(a_np).cuda()
                b = torch.from_numpy(b_np).cuda()
            else:
                a, b = a_np, b_np
            
            # Verify correctness first
            result = fn(a, b)
            if not self.verify_result(result, reference, name):
                print(f"Skipping {name} benchmark due to verification failure")
                continue
            
            # Warmup
            for _ in range(warmup):
                _ = fn(a, b)
                if name.startswith("cuda") or name.startswith("torch"):
                    torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(rep):
                start = time.perf_counter()
                _ = fn(a, b)
                if name.startswith("cuda") or name.startswith("torch"):
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            
            # Calculate statistics
            times = np.array(times)
            self.results[f"{name}_{M}x{N}x{K}"] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "tflops": (2 * M * N * K) / (np.mean(times) * 1e12),
                "verified": True
            }
    
    def run_all_benchmarks(self):
        """Run all implementations."""
        # 1. NumPy (CPU baseline)
        self.benchmark_fn(
            lambda a, b: np.matmul(a, b),
            "numpy_cpu"
        )
        
        # 2. cuBLAS
        self.benchmark_fn(
            lambda a, b: torch.matmul(a, b),
            "cublas"
        )
        
        # 3. Custom CUDA kernel
        self.benchmark_fn(
            lambda a, b: custom_cuda_matmul(a, b), # type: ignore
            "custom_cuda"
        )
        
        # 4. Triton kernel
        self.benchmark_fn(
            lambda a, b: triton_matmul(a, b), # type: ignore
            "triton"
        )
    
    def print_results(self):
        """Pretty print results with comparison to baseline."""
        # Find baseline (NumPy CPU) results for each size
        baselines = {
            size: next(v for k, v in self.results.items() 
                      if k.startswith("numpy_cpu") and size in k)
            for size in [f"{M}x{N}x{K}" for M, N, K in self.sizes]
        }
        
        # Print results grouped by size
        for size in baselines.keys():
            print(f"\nResults for size {size}:")
            print(f"{'Implementation':<15} {'Time (ms)':<10} {'Speedup':<10} {'TFLOPS':<10}")
            print("-" * 45)
            
            baseline_time = baselines[size]["mean"]
            
            for name, stats in self.results.items():
                if size not in name:
                    continue
                    
                impl_name = name.split("_")[0]
                speedup = baseline_time / stats["mean"]
                
                print(f"{impl_name:<15} "
                      f"{stats['mean']*1000:>9.2f} "
                      f"{speedup:>9.1f}x "
                      f"{stats['tflops']:>9.1f}")

# Example usage:
if __name__ == "__main__":
    sizes = [
        (1024, 1024, 1024),    # Standard square
        (4096, 4096, 4096),    # Large square
        (8192, 128, 1024),     # Typical attention shape
    ]
    
    benchmark = KernelBenchmark(sizes)
    benchmark.run_all_benchmarks()
    benchmark.print_results()