// cudnn_utils.cu
#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>

"""
    Apply layer normalization: y = γ * ((x - μ) / √(σ² + ε)) + β
    
    Args:
        x: Input tensor (batch_size, hidden_dim) - must be contiguous GPU tensor
        gamma: Scale parameter (hidden_dim,) - must be contiguous GPU tensor
        beta: Shift parameter (hidden_dim,) - must be contiguous GPU tensor
        eps: Epsilon for numerical stability
        handle: Existing cuDNN handle (optional)
"""

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
    float* mean,        // mean buffer
    float* variance,    // variance buffer
    int batch_size,     // N
    int hidden_dim,     // C
    float eps          // epsilon
) {
    // Create handle
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    
    // Create tensor descriptor
    cudnnTensorDescriptor_t x_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    
    // Set descriptor - NCHW format [batch_size, hidden_dim, 1, 1]
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NCHW,     // Format
        CUDNN_DATA_FLOAT,      // Data type
        batch_size,            // N
        hidden_dim,            // C
        1,                     // H
        1                      // W
    ));
    
    // Forward pass
    CHECK_CUDNN(cudnnLayerNormalizationForward(
        handle,
        CUDNN_LAYER_NORM_FORWARD_INFERENCE,  // Mode
        CUDNN_LAYER_NORM_PER_CHANNEL,        // Normalization mode
        x,                                   // Input data
        x_desc,                              // Input descriptor
        output,                              // Output data
        x_desc,                              // Output descriptor
        gamma,                               // Scale parameter
        beta,                                // Shift parameter
        eps,                                 // Epsilon
        mean,                                // Mean buffer
        variance                             // Variance buffer
    ));
    
    // Cleanup
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroy(handle);
    
    return CUDNN_STATUS_SUCCESS;
}

// Helper function to get workspace size if needed
size_t get_layernorm_workspace_size(
    int batch_size,
    int hidden_dim
) {
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        hidden_dim,
        1, 1
    );
    
    size_t workspace_size = 0;
    cudnnGetLayerNormalizationWorkspaceSize(
        handle,
        CUDNN_LAYER_NORM_FORWARD_INFERENCE,
        CUDNN_LAYER_NORM_PER_CHANNEL,
        x_desc,
        &workspace_size
    );
    
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroy(handle);
    
    return workspace_size;
}

} // extern "C"