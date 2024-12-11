import torch
import numpy as np
import cupy.cuda.cublas as cublas
import ctypes

def cudnn_layernorm(x, gamma, beta, eps = 1e-5):
    """
    Apply layer normalization: y = γ * ((x - μ) / √(σ² + ε)) + β
    
    Args:
        x: Input tensor (batch_size, hidden_dim) - must be contiguous GPU tensor
        gamma: Scale parameter (hidden_dim,) - must be contiguous GPU tensor
        beta: Shift parameter (hidden_dim,) - must be contiguous GPU tensor
        eps: Epsilon for numerical stability
        handle: Existing cuDNN handle (optional)
    """
    # Create or use existing handle
    should_destroy = False
    if handle is None:
        handle = cudnn.cudnnCreate()
        should_destroy = True
        
    try:
        # Create descriptor for input/output tensor
        x_desc = cudnn.cudnnCreateTensorDescriptor()
        
        # Configure descriptor
        # NCHW format: [batch_size, channels, height, width]
        # For LayerNorm, we reshape to: [N, C, 1, 1]
        cudnn.cudnnSetTensor4dDescriptor(
            x_desc,
            cudnn.CUDNN_TENSOR_NCHW,     # Format
            cudnn.CUDNN_DATA_FLOAT,      # Data type
            x.shape[0],                  # Batch size
            x.shape[1],                  # Hidden dim
            1, 1                         # Height, Width = 1
        )
        
        # Allocate buffers for mean and variance
        mean = torch.zeros_like(x)
        variance = torch.zeros_like(x)
        
        # Forward pass
        cudnn.cudnnLayerNormForward(
            handle,
            x_desc,                      # Input descriptor
            x,                           # Input data
            x_desc,                      # Output descriptor
            output,                      # Output buffer
            gamma,                       # Scale parameter
            beta,                        # Shift parameter
            eps,                         # Epsilon
            mean,                        # Mean buffer
            variance                     # Variance buffer
        )
        
    finally:
        # Clean up
        cudnn.cudnnDestroyTensorDescriptor(x_desc)
        if should_destroy:
            cudnn.cudnnDestroy(handle)