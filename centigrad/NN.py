# Neural Network implementation of the centigrad engine. 
# In the actual transformer model, we have a slightly different setup for the MLP.

import numpy as np
import random
from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):

    def __init__(self, in_features, out_features):
        self.weight = Value(np.random.randn(in_features, out_features) * 0.02)
        self.bias = Value(np.zeros((1, out_features)))
    
    def forward(self, x):
        return x.matmul(self.weight) + self.bias
    
    def parameters(self):
        return [self.weight, self.bias]

class LayerNorm(Module):

    def __init__(self, dim):
        self.gamma = Value(np.ones(dim))
        self.beta = Value(np.zeros(dim))

    def forward(self, x):
        return x.layer_norm(self.gamma, self.beta)
    
    def parameters(self):
        return [self.gamma, self.beta]
    
class SwiGLU(Module):
    def __init__(self, dim, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.w_proj = Linear(dim, hidden_dim)
        self.v_proj = Linear(dim, hidden_dim)
    
    def forward(self, x):
        return self.w_proj(x) * self.v_proj(x).swish()
    
    def parameters(self):
        return self.w_proj.parameters() + self.v_proj.parameters()
    
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Value(np.random.randn(num_embeddings, embedding_dim) * 0.02)
    
    def forward(self, indices):
        # This uses the existing embedding method from Value class
        return self.weight.embedding(indices, self.weight)
    
    def parameters(self):
        return [self.weight]
# All code below is NOT implemented in the transformer

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.v = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.c = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        # First linear transformation
        act1 = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # Second linear transformation
        act2 = sum((vi * xi for vi, xi in zip(self.v, x)), self.c)
        # SwiGLU = (Wx + b) * swish(Vx + c)
        out = act1 * act2.swish()
        return out
    
    def parameters(self):
        return self.w + [self.b] + self.v + [self.c]
    
    def __repr__(self):
        return f"{'SwiGLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, max_grad_norm=1.0):
        """
        Args:
            params: iterable of parameters to optimize
            lr: learning rate
            betas: coefficients for moving averages (b1, b2)
            eps: term for numerical stability
            weight_decay: weight decay (L2 penalty)
        """
       
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        # Initialize momentum and velocity terms for each parameter
        self.m = [np.zeros_like(p.data) for p in self.params]  # First moment
        self.v = [np.zeros_like(p.data) for p in self.params]  # Second moment
        self.t = 0  # Time step for bias correction

    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def clip_grad_norm(self):
        """Clips gradient norm of parameters."""
        total_norm = 0
        for p in self.params:
            if p.grad is not None:
                param_norm = np.linalg.norm(p.grad.flatten())
                total_norm += param_norm ** 2
        total_norm = np.sqrt(total_norm)
        
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.params:
                if p.grad is not None:
                    p.grad *= clip_coef
        
        return total_norm

    def step(self):
        """Performs a single optimization step with gradient clipping."""
        # First clip gradients
        grad_norm = self.clip_grad_norm()
        
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
                
            # Apply weight decay
            p.data -= self.lr * self.weight_decay * p.data
            
            # Update momentum terms
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
        return grad_norm  # Return the gradient norm for logging

class DataLoader:
    def  __init__(self, data, batch_size, seq_length, split='train'):
        """
        Args:
            data: array of token ids
            batch_size: number of sequences per batch
            seq_length: length of each sequences
            split: 'train' or 'val' to handle differently
        """

        self.data = data
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.split = split

        # Calculate number of batches
        self.n_batches = len(data) // (batch_size * seq_length)

        # Initialize position
        self.current_pos = 0

    def __iter__(self):
        if self.split == 'train':  # Reset position at start of each epoch for training
            self.current_pos = 0
    
    def get_batch(self):
        # Get chunk of data and targets
        chunk_size = self.batch_size * self.seq_length
        data_chunk = self.data[self.current_pos:self.current_pos + chunk_size + 1]
        
        # Update position and handle wrap-around
        self.current_pos += chunk_size
        if self.current_pos + chunk_size + 1 > len(self.data):
            self.current_pos = 0
            
        # Reshape into batches
        x = np.array(data_chunk[:-1]).reshape(self.batch_size, self.seq_length)
        y = np.array(data_chunk[1:]).reshape(self.batch_size, self.seq_length)
        
        # Convert to Value objects
        return Value(x), Value(y)