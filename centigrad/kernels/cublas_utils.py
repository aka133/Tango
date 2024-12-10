import torch
import numpy as np
from cuda import cublas
import ctypes

def cublas_matmul(A, B, C, M, N, K):
    """
    Compute C = A @ B using cuBLAS
    A: (M, K)
    B: (K, N)
    C: (M, N)
    """

    # Get handle to cuBLAS context
    handle = cublas.cublasCreate()

    # Set matrix operation parameters
    alpha = ctypes.c_float(1.0)

    # Set matrix operation parameters
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    # Call cuBLAS SGEMM (single precision general matrix multiply)
    cublas.cublasSgemm(
        handle,
        cublas.CUBLAS_OP_N,
        cublas.CUBLAS_OP_N,
        M, N, K,
        alpha,
        A, M, # A matrix, leading dimension
        B, K, # B matrix, leading dimension
        beta,
        C, M # C matrix, leading dimension
    )

    # Clean up
    cublas.cublasDestroy(handle)