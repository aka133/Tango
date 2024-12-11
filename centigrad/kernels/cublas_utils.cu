#include <cublas_v2.h>

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

extern "C" {
    void cublas_matmul(float* A, float* B, float* C,
                        int M, int N, int K,
                        float alpha, float beta) {
        
        cublasHandle_t handle; 
        cublasCreate(&handle);

        cublasSgemm(handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    M, N, K, 
                    &alpha, 
                    A, M, 
                    B, K, 
                    &beta, 
                    C, M);

        cublasDestroy(handle);
    }
}