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