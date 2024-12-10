import torch
import numpy as np
from cuda import cublas
import ctypes

def cudnn_layernorm(x, gamma, beta, eps = 1e-5):
    """
    Apply layer normalization using cuDNN
    x: input tensor
    gamma, beta: scale and shift parameters
    """

    # Get cuDNN handle
    handle = cudnn.cudnnCreate()

    # Create tensor descriptors
    x_desc = cudnn.cudnnCreateTensorDescriptor()
    # Set descriptor for input
    cudnn.cudnnSetTensor4dDescriptor(
        x_desc,
        cudnn.CUDNN_TENSOR_NCHW,
        cudnn.CUDNN_DATA_FLOAT,
        *x.shape
    )

    # Call cuDNN LayerNorm
    cudnn.cudnnLayerNormForward(
        handle,
        x_desc, # Input descriptor
        x, # Input tensor
        x_desc, # Output descriptor
        output, # Output tensor
        gamma, # Scale parameter
        beta, # Shift parameter
        eps, # Epsilon for numerical stability
        mean, # Mean buffer (optional)
        variance # Variance buffer (optional)
    )

    # Clean up
    cudnn.cudnnDestroyTensorDescriptor(x_desc)
    cudnn.cudnnDestroy(handle)