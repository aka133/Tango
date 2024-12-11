import torch
import numpy as np
import cupy.cuda.cublas as cublas

def cublas_matmul(A, B, C, M, N, K, alpha=1.0, beta=0.0, handle=None):
    """
    Compute: C = α(A @ B) + βC using cuBLAS SGEMM
    
    Args:
        A: Input matrix (M, K) - must be contiguous GPU tensor
        B: Input matrix (K, N) - must be contiguous GPU tensor
        C: Output matrix (M, N) - must be contiguous GPU tensor
        M, N, K: Matrix dimensions
        alpha: Scalar for AB product (default: 1.0)
        beta: Scalar for C (default: 0.0)
        handle: Existing cuBLAS handle (optional)
    """

    # Create or use existing handle
    should_destroy = False
    if handle is None:
        handle = cublas.cublasCreate()
        should_destroy = True

    try:

        # Convert scalars to ctypes for cuBLAS
        alpha_c = ctypes.c_float(alpha)
        beta_c = ctypes.c_float(beta)
        
        # SGEMM: Single-precision General Matrix Multiply
        # C = α(A @ B) + βC
        # Note: cuBLAS uses column-major order!

        cublas.cublasSgemm(
            handle,
            cublas.CUBLAS_OP_N,
            cublas.CUBLAS_OP_N,
            M, N, K,
            alpha_c,
            A, M, # A matrix, leading dimension
            B, K, # B matrix, leading dimension
            beta_c,
            C, M # C matrix, leading dimension
        )

    finally:
        # Clean up
        if should_destroy:
            cublas.cublasDestroy(handle)
