// cudnn_utils.cu
#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>

/*
    Apply layer normalization: y = γ * ((x - μ) / √(σ² + ε)) + β
    
    Args:
        x: Input tensor (batch_size, hidden_dim) - must be contiguous GPU tensor
        gamma: Scale parameter (hidden_dim,) - must be contiguous GPU tensor
        beta: Shift parameter (hidden_dim,) - must be contiguous GPU tensor
        eps: Epsilon for numerical stability
        handle: Existing cuDNN handle (optional)
*/

extern "C" {

// Error checking helper
#define CHECK_CUDNN(call) do {                                \
    cudnnStatus_t status = call;                             \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
        printf("CUDNN error: %s\n", cudnnGetErrorString(status)); \
        return status;                                       \
    }                                                        \
} while(0)

cudnnStatus_t cudnn_layernorm(
    float* x,           // input data
    float* gamma,       // scale parameter
    float* beta,        // shift parameter
    float* output,      // output buffer
    float* mean,        // mean buffer (optional, can be NULL for inference)
    float* variance,    // variance buffer (optional, can be NULL for inference)
    int batch_size,     // N
    int hidden_dim,     // C
    float eps,         // epsilon
    bool is_training   // whether in training mode
) {
    // Create handle
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    
    // Create tensor descriptor
    cudnnTensorDescriptor_t tensor_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensor_desc));
    
    // Set descriptor - NCHW format [batch_size, hidden_dim, 1, 1]
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        tensor_desc,
        CUDNN_TENSOR_NCHW,  // Format
        CUDNN_DATA_FLOAT,   // Data type
        batch_size,         // N
        hidden_dim,         // C
        1,                  // H
        1                   // W
    ));
    
    // Parameters for batch normalization
    // To emulate layer normalization, set spatial dimensions to 1
    const float alpha = 1.0f;
    const float beta_val = 0.0f;
    const double exp_avg_factor = 1.0;
    const double epsilon = static_cast<double>(eps);

    if (is_training) {
        // Forward training pass
        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta_val,
            tensor_desc,
            x,
            tensor_desc,
            output,
            tensor_desc,
            gamma,
            beta,
            exp_avg_factor,
            mean,
            variance,
            epsilon,
            mean,
            variance
        ));
    } else {
        // Inference pass 
        CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta_val,
            tensor_desc,
            x,
            tensor_desc,
            output,
            tensor_desc,
            gamma,
            beta,
            epsilon,
            mean,
            variance
        ));
    }
    
    // Cleanup
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensor_desc));
    CHECK_CUDNN(cudnnDestroy(handle));
    
    return CUDNN_STATUS_SUCCESS;
}

// Helper function to get workspace size
size_t get_layernorm_workspace_size(
    int batch_size,
    int hidden_dim
) {
    return 0;  // No workspace needed
}

} // extern "C"