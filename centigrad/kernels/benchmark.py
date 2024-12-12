import subprocess

matrix_sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]

for M, N, K in matrix_sizes:
    print(f"\nBenchmarking size {M}x{N}x{K}")
    subprocess.run(["python", "cuda_benchmark.py", str(M), str(N), str(K)])
    subprocess.run(["python", "cublas_benchmark.py", str(M), str(N), str(K)])
    subprocess.run(["python", "triton_benchmark.py", str(M), str(N), str(K)])
    subprocess.run(["python", "torch_benchmark.py", str(M), str(N), str(K)])