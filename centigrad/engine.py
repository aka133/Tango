import math
import numpy as np

class Value:
    ##Preserve most code from Karpathy's micrograd
    def __init__(self, data, _children=(), _op = ''):
        self.data = np.array(data)
        self._prev = set(_children)
        self._op = _op
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1
        out._backward = _backward

        return out
    
    def __mult__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), '**')

        def _backward():
            self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
            other.grad += (math.log(self.data) * self.data ** other.data) * out.grad
        out._backward = _backward

        return out
    
    def __neg__(self):
        return (-1) * self
    
    def __radd__(self, other): 
        return self + other
    
    def __sub__(self, other): 
        return self + (-other)
    
    def __rsub__(self, other): 
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other): 
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    ##Define SwiGLU activation function
    def sigmoid(self):
        x = self.data
        out = Value(1 / (1 + math.exp(-x)), (self, ), 'sigmoid')

        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward
        
        return out
    
    def swish(self):
        x = self.data
        sig = self.sigmoid()
        out = Value(x * sig.data, (self, ), 'swish')

        def _backward():
            self.grad += (sig.data + x * sig.data * (1 - sig.data)) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()