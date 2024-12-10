import triton
import triton.language as tl

@triton.jit

def primitive_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, 
    # Block sizes (known at compile time)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute: C = A @ B
    """

    # Program ID
    pid = tl.program_id(0)

    # Block ID
    # Note: Triton automatically handles shared memory and threading
    block_m = pid // (N // BLOCK_SIZE_N)
    block_n = pid % (N // BLOCK_SIZE_N)

    # Offses to the start of the block
    offs_am = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    # Note: Triton handles register allocation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate through K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks from A and B
        a = tl.load(a_ptr + offs_am[:, None] * K + (k + offs_k[None, :]))
        b = tl.load(b_ptr + (k + offs_k[:, None]) * N + offs_bn[None, :])

        # Compute block-block dot product
        acc += tl.dot(a, b)

    # Write result
    c = acc.to(tl.float32)
    offs_cm = offs_am
    offs_cn = offs_bn
    tl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :], c)

# Grid and block configuration
@triton.autotune(
    configs = [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}),
    ],
    key = ['M', 'N', 'K']
)

@triton.jit

# Optimized kernel with improved autotuning parameters and block index calculation
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, 
    # Block sizes (known at compile time)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Optional: Memory layout hints
    IS_A_ROW_MAJOR: tl.constexpr = True,
    IS_B_ROW_MAJOR: tl.constexpr = True,
):
    """
    Optimized matrix multiplication C = A @ B
    """
    pid = tl.program_id(0)
    
    # Improved block index calculation
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_n
    group_id = pid // num_pid_in_group
    pid_n = pid % num_pid_in_group
    pid_m = group_id

    # Offset calculations
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator with proper dtype
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop
    for k in range(0, K, BLOCK_SIZE_K):
        # Smart pointer arithmetic based on layout
        if IS_A_ROW_MAJOR:
            a = tl.load(a_ptr + offs_am[:, None] * K + (k + offs_k[None, :]))
        else:
            a = tl.load(a_ptr + (k + offs_k[None, :]) * M + offs_am[:, None])
            
        if IS_B_ROW_MAJOR:
            b = tl.load(b_ptr + (k + offs_k[:, None]) * N + offs_bn[None, :])
        else:
            b = tl.load(b_ptr + offs_bn[None, :] * K + (k + offs_k[:, None]))

        # Use tensor cores when possible
        acc += tl.dot(a, b, allow_tf32=True)

    # Write back result
    c = acc.to(tl.float32)
    offs_cm = offs_am
    offs_cn = offs_bn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :], c, mask=mask)

@triton.autotune(
    configs=[
        # Let autotuner try different configurations
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}),
    ],
    key = ['M', 'N', 'K']
)

def matmul1(A, B, C, M, N, K):
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    primitive_matmul_kernel[grid](A, B, C, M, N, K, **META)

def matmul2(A, B, C, M, N, K):
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](A, B, C, M, N, K, **META)