import cupy as cp

# Load all kernels from the .cu file
with open('centigrad/kernels/cuda_kernels.cu', 'r') as f:
    cuda_source = f.read()

# Create raw kernels
primitive_matmul_kernel = cp.RawKernel(cuda_source, 'primitive_matmul')
primitive_coalesced_matmul_kernel = cp.RawKernel(cuda_source, 'primitive_coalesced_matmul')
float4_coalesced_matmul_kernel = cp.RawKernel(cuda_source, 'float4_coalesced_matmul')
add_loop_unrolling_matmul_kernel = cp.RawKernel(cuda_source, 'add_loop_unrolling_matmul')
double_buffering_loop_unrolling_matmul_kernel = cp.RawKernel(cuda_source, 'double_buffering_loop_unrolling_matmul')

# Simple wrapper functions
def run_kernel(kernel, a, b):
    M, K = a.shape
    _, N = b.shape
    c = cp.zeros((M, N), dtype=cp.float32)
    
    block_dim = (32, 32)
    grid_dim = ((N + 31) // 32, (M + 31) // 32)
    
    kernel(grid_dim, block_dim, (a, b, c, M, N, K))
    return c

# Export these simple wrappers
def primitive_matmul(a, b):
    return run_kernel(primitive_matmul_kernel, a, b)

def primitive_coalesced_matmul(a, b):
    return run_kernel(primitive_coalesced_matmul_kernel, a, b)

def float4_coalesced_matmul(a, b):
    return run_kernel(float4_coalesced_matmul_kernel, a, b)

def add_loop_unrolling_matmul(a, b):
    return run_kernel(add_loop_unrolling_matmul_kernel, a, b)

def double_buffering_loop_unrolling_matmul(a, b):
    return run_kernel(double_buffering_loop_unrolling_matmul_kernel, a, b)