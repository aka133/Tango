#include <vector_types.h>

#ifdef __CUDACC__
#define ALIGN(x) __align__(x)
#else
#define ALIGN(x)
#endif

extern "C" {

__global__ void primitive_matmul(float* A, float* B, float* C, int M, int N, int K) {
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (row < M && col < N) {
        float sum = 0.0f;
        // Loop over K dimension
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        // Write result to global memory
        C[row * N + col] = sum;
    }
}

// Coalesced memory for matmul

__global__ void primitive_coalesced_matmul(float* A, float *B, float *C, int M, int N, int K) {
    // Define shared memory arrays for A and B tiles
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize sum to 0
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + 31) / 32; t++) {
        // Load tiles into shared memory
        if (row < M && t * 32 + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * 32 + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (col < N && t * 32 + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * 32 + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize threads to ensure tiles are loaded
        __syncthreads();

        // Compute partial results
        for (int k = 0; k < 32; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize threads
        __syncthreads();

    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Use float4 for memory tiling to speed up memory access

__global__ void float4_coalesced_matmul(float* A, float* B, float* C, int M, int N, int K) {
    // Define shared memory arrays for A and B tiles
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize sum to 0
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + 31) / 32; t++) {
        // Load tiles into shared memory using float4
        if (row < M && (t * 32 + threadIdx.x * 4) < K) {
            float4 a = reinterpret_cast<float4*>(A + row * K + t * 32 + threadIdx.x * 4)[0];
            tileA[threadIdx.y][threadIdx.x * 4 + 0] = a.x;
            tileA[threadIdx.y][threadIdx.x * 4 + 1] = a.y;
            tileA[threadIdx.y][threadIdx.x * 4 + 2] = a.z;
            tileA[threadIdx.y][threadIdx.x * 4 + 3] = a.w;
        } else {
            tileA[threadIdx.y][threadIdx.x * 4 + 0] = 0.0f;
            tileA[threadIdx.y][threadIdx.x * 4 + 1] = 0.0f;
            tileA[threadIdx.y][threadIdx.x * 4 + 2] = 0.0f;
            tileA[threadIdx.y][threadIdx.x * 4 + 3] = 0.0f;
        }
        if (col < N && (t * 32 + threadIdx.y * 4) < K) {
            float4 b = reinterpret_cast<float4*>(B + (t * 32 + threadIdx.y * 4) * N + col)[0];
            tileB[threadIdx.y * 4 + 0][threadIdx.x] = b.x;
            tileB[threadIdx.y * 4 + 1][threadIdx.x] = b.y;
            tileB[threadIdx.y * 4 + 2][threadIdx.x] = b.z;
            tileB[threadIdx.y * 4 + 3][threadIdx.x] = b.w;
        } else {
            tileB[threadIdx.y * 4 + 0][threadIdx.x] = 0.0f;
            tileB[threadIdx.y * 4 + 1][threadIdx.x] = 0.0f;
            tileB[threadIdx.y * 4 + 2][threadIdx.x] = 0.0f;
            tileB[threadIdx.y * 4 + 3][threadIdx.x] = 0.0f;
        }

        // Synchronize threads to ensure tiles are loaded
        __syncthreads();

        // Compute partial results
        for (int k = 0; k < 32; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize threads
        __syncthreads();

    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Add loop unrolling to the coalesced memory matmul and add a column buffer to prevent bank conflicts

__global__ void add_loop_unrolling_matmul(float* A, float* B, float* C, int M, int N, int K) {
    // Define shared memory arrays for A and B tiles
    __shared__ float tileA[32][33];
    __shared__ float tileB[32][33];

    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize sum to 0
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + 31) / 32; t++) {
        // Load tiles into shared memory using float4
        if (row < M && (t * 32 + threadIdx.x * 4) < K) {
            float4 a = reinterpret_cast<float4*>(A + row * K + t * 32 + threadIdx.x * 4)[0];
            tileA[threadIdx.y][threadIdx.x * 4 + 0] = a.x;
            tileA[threadIdx.y][threadIdx.x * 4 + 1] = a.y;
            tileA[threadIdx.y][threadIdx.x * 4 + 2] = a.z;
            tileA[threadIdx.y][threadIdx.x * 4 + 3] = a.w;
        } else {
            tileA[threadIdx.y][threadIdx.x * 4 + 0] = 0.0f;
            tileA[threadIdx.y][threadIdx.x * 4 + 1] = 0.0f;
            tileA[threadIdx.y][threadIdx.x * 4 + 2] = 0.0f;
            tileA[threadIdx.y][threadIdx.x * 4 + 3] = 0.0f;
        }
        
        if (col < N && (t * 32 + threadIdx.y * 4) < K) {
            float4 b = reinterpret_cast<float4*>(B + (t * 32 + threadIdx.y * 4) * N + col)[0];
            tileB[threadIdx.y * 4 + 0][threadIdx.x] = b.x;
            tileB[threadIdx.y * 4 + 1][threadIdx.x] = b.y;
            tileB[threadIdx.y * 4 + 2][threadIdx.x] = b.z;
            tileB[threadIdx.y * 4 + 3][threadIdx.x] = b.w;
        } else {
            tileB[threadIdx.y * 4 + 0][threadIdx.x] = 0.0f;
            tileB[threadIdx.y * 4 + 1][threadIdx.x] = 0.0f;
            tileB[threadIdx.y * 4 + 2][threadIdx.x] = 0.0f;
            tileB[threadIdx.y * 4 + 3][threadIdx.x] = 0.0f;
        }

        // Synchronize threads to ensure tiles are loaded
        __syncthreads();

        // Multiple accumulators (parallel paths):
        float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;

        #pragma unroll 8
        for (int k = 0; k < 32; k += 4) {
            // These four operations can execute in parallel:
            sum1 += tileA[threadIdx.y][k+0] * tileB[k+0][threadIdx.x];
            sum2 += tileA[threadIdx.y][k+1] * tileB[k+1][threadIdx.x];
            sum3 += tileA[threadIdx.y][k+2] * tileB[k+2][threadIdx.x];
            sum4 += tileA[threadIdx.y][k+3] * tileB[k+3][threadIdx.x];
        }

        // Combine at the end
        sum = sum1 + sum2 + sum3 + sum4;

        // Synchronize threads
        __syncthreads();

    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// Double buffering to save time for memory access


__global__ void double_buffering_loop_unrolling_matmul(float* A, float* B, float* C, int M, int N, int K) {
    // Define shared memory arrays for A and B tiles
    __shared__ float tileA[2][32][33];
    __shared__ float tileB[2][32][33];

    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize sum to 0
    float sum = 0.0f;

    // Load first tiles into buffer 0

    if (row < M && (threadIdx.x * 4) < K) {
        float4 a = reinterpret_cast<float4*>(A + row * K + threadIdx.x * 4)[0];
        tileA[0][threadIdx.y][threadIdx.x * 4 + 0] = a.x;
        tileA[0][threadIdx.y][threadIdx.x * 4 + 1] = a.y;
        tileA[0][threadIdx.y][threadIdx.x * 4 + 2] = a.z;
        tileA[0][threadIdx.y][threadIdx.x * 4 + 3] = a.w;
    }

    if (col < N && (threadIdx.y * 4) < K) {
        float4 b = reinterpret_cast<float4*>(B + (threadIdx.y * 4) * N + col)[0];
        tileB[0][threadIdx.y * 4 + 0][threadIdx.x] = b.x;
        tileB[0][threadIdx.y * 4 + 1][threadIdx.x] = b.y;
        tileB[0][threadIdx.y * 4 + 2][threadIdx.x] = b.z;
        tileB[0][threadIdx.y * 4 + 3][threadIdx.x] = b.w;
    }

    // Need sync after initial load
    __syncthreads();

    // Loop over tiles
    for (int t = 0; t < (K + 31) / 32; t++) {
        int current_buffer = t % 2;
        int next_buffer = (t + 1) % 2;

        // Load next tile (t+1) while computing current tile (t)
        if (t + 1 < (K + 31) / 32) {  // Check if there's a next tile
            if (row < M && ((t+1) * 32 + threadIdx.x * 4) < K) {
                float4 a = reinterpret_cast<float4*>(A + row * K + (t+1) * 32 + threadIdx.x * 4)[0];
                tileA[next_buffer][threadIdx.y][threadIdx.x * 4 + 0] = a.x;
                tileA[next_buffer][threadIdx.y][threadIdx.x * 4 + 1] = a.y;
                tileA[next_buffer][threadIdx.y][threadIdx.x * 4 + 2] = a.z;
                tileA[next_buffer][threadIdx.y][threadIdx.x * 4 + 3] = a.w;
            } else {
                tileA[next_buffer][threadIdx.y][threadIdx.x * 4 + 0] = 0.0f;
                tileA[next_buffer][threadIdx.y][threadIdx.x * 4 + 1] = 0.0f;
                tileA[next_buffer][threadIdx.y][threadIdx.x * 4 + 2] = 0.0f;
                tileA[next_buffer][threadIdx.y][threadIdx.x * 4 + 3] = 0.0f;
            }
            if (col < N && ((t+1) * 32 + threadIdx.y * 4) < K) {
                float4 b = reinterpret_cast<float4*>(B + ((t+1) * 32 + threadIdx.y * 4) * N + col)[0];
                tileB[next_buffer][threadIdx.y * 4 + 0][threadIdx.x] = b.x;
                tileB[next_buffer][threadIdx.y * 4 + 1][threadIdx.x] = b.y;
                tileB[next_buffer][threadIdx.y * 4 + 2][threadIdx.x] = b.z;
                tileB[next_buffer][threadIdx.y * 4 + 3][threadIdx.x] = b.w;
            } else {
                tileB[next_buffer][threadIdx.y * 4 + 0][threadIdx.x] = 0.0f;
                tileB[next_buffer][threadIdx.y * 4 + 1][threadIdx.x] = 0.0f;
                tileB[next_buffer][threadIdx.y * 4 + 2][threadIdx.x] = 0.0f;
                tileB[next_buffer][threadIdx.y * 4 + 3][threadIdx.x] = 0.0f;
            }
        }  

        // Multiple accumulators (parallel paths):
        float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;

        #pragma unroll 8
        for (int k = 0; k < 32; k += 4) {
            // These four operations can execute in parallel:
            sum1 += tileA[current_buffer][threadIdx.y][k+0] * tileB[current_buffer][k+0][threadIdx.x];
            sum2 += tileA[current_buffer][threadIdx.y][k+1] * tileB[current_buffer][k+1][threadIdx.x];
            sum3 += tileA[current_buffer][threadIdx.y][k+2] * tileB[current_buffer][k+2][threadIdx.x];
            sum4 += tileA[current_buffer][threadIdx.y][k+3] * tileB[current_buffer][k+3][threadIdx.x];
        }

        // Combine at the end
        sum = sum1 + sum2 + sum3 + sum4;

        // Synchronize threads
        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Wrapper functions with proper launch configuration
void launch_float4_matmul(float* A, float* B, float* C, int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    float4_coalesced_matmul<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_double_buffer_matmul(float* A, float* B, float* C, int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    double_buffering_loop_unrolling_matmul<<<grid, block>>>(A, B, C, M, N, K);
}

} // extern "C"