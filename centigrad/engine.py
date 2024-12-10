import math
import numpy as np

class Value:
    # Preserve most code from Karpathy's micrograd
    def __init__(self, data, _children=(), _op = ''):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.device = 'cpu' # GPU support coming soon
        self._prev = set(_children)
        self._op = _op
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, shape={self.shape})"
   
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1
        out._backward = _backward

        return out
    
    def __mul__(self, other):
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
            other.grad += (np.log(self.data) * self.data ** other.data) * out.grad
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
        out = Value(np.exp(self.data), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    # Add support for more tensor operations

    def matmul(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(np.matmul(self.data, other.data), (self, other), 'matmul')

        def _backward():
            self.grad += np.matmul(out.grad, other.data.T)
            other.grad += np.matmul(self.data.T, out.grad)
        out._backward = _backward

        return out
    
    def reshape(self, *shape):
        """Reshape tensor to new shape.
        Example: if x has shape (6,)
        x.reshape(2,3) will have shape (2,3)
        """
        out = Value(self.data.reshape(shape), (self,), 'reshape')

        def _backward():
            # If we went from shape (6,) to (2,3)
            # Gradients need to go from (2,3) back to (6,)
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward

        return out

    def contiguous(self):
        return self
    
    def gather(self, dim, indices):

        out = Value(np.take_along_axis(self.data, indices, dim), (self,), 'gather')

        def _backward():
            # Gradients from gather are sparse, so we need to sum them back to dense
            np.add.at(self.grad, tuple(indices), out.grad)
        out._backward = _backward

        return out

    def split(self, dim, indices):
        """Split tensor along specified axis at given indices.
        Example: if x has shape (2,3,4), split(axis=1, indices=[1])
        will return two tensors with shapes (2,1,4) and (2,2,4)
        """
        chunks = np.split(self.data, indices, dim)
        out = [Value(chunk, (self,), 'split') for chunk in chunks]

        def _backward():
            self.grad += np.concatenate([chunk.grad for chunk in out], axis=dim)

        for chunk in out:
            chunk._backward = _backward

        return out

    def topk(self, k, dim =-1):
       
        indices = np.argsort(self.data, dim)[..., k:]

        values = np.take_along_axis(self.data, indices, dim)
        out = Value(values, (self,), 'topk')

        def _backward():
            np.add.at(self.grad, indices, out.grad)
        out._backward = _backward

        return out

    def transpose(self, dim0, dim1):
        """Swap two dimensions of tensor.
        Example: if x has shape (2,3,4)
        x.transpose(0,1) will have shape (3,2,4)
        """
        dims = list(range(len(self.shape)))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        out = Value(self.data.transpose(dims), (self,), 'transpose')

        def _backward():
            # Use same dimension swap to get gradients back to original shape
            # If we swapped dims (0,1) forward, we need to swap (0,1) backward
            self.grad = out.grad.transpose(dims)
        out._backward = _backward

        return out

    def scaled_dot_product_attention(self, q, k, v, is_causal=False):
        dk = q.shape[-1]
        scaling_factor = 1 / np.sqrt(dk)
        attn = (q @ k.transpose(-2, -1)) * scaling_factor

        if is_causal:
            mask = attn.causal_mask(attn.shape[-2])
            attn.data = np.where(mask, float('-inf'), attn.data)

        attn = attn.softmax(dim=-1)

        out = attn @ v

        def _backward():
            self.grad += [out.grad @ v.transpose(-2, -1)] * scaling_factor
            v.grad += attn.transpose @ out.grad
            q.grad += (attn.grad @ k) * scaling_factor
            k.grad += (attn.grad @ k) * scaling_factor
        
        out._backward = _backward

        return out

    def softmax(self, dim=-1):
        """Softmax along specified dimension.
        Example: if x has shape (2,3), softmax(dim=1) operates on each row
        """
        # Subtract max for numerical stability
        x_max = np.max(self.data, axis=dim, keepdims=True)
        exp_x = np.exp(self.data - x_max)
        # Sum along dimension and keep dims for division
        sum_exp_x = np.sum(exp_x, axis=dim, keepdims=True)
        out = Value(exp_x / sum_exp_x, (self,), 'softmax')

        def _backward():
            # Softmax gradient: softmax_i * (kronecker_ij - softmax_j) * grad_i
            s = out.data
            grad_times_s = out.grad * s
            sum_grad_times_s = np.sum(grad_times_s, axis=dim, keepdims=True)
            self.grad += s * (out.grad - sum_grad_times_s)
        out._backward = _backward

        return out
    
    def log_softmax(self, dim=-1):
        """Log softmax along specified dimension."""
       
        # Subtract max for numerical stability
        x_max = np.max(self.data, axis=dim, keepdims=True)
        exp_x = np.exp(self.data - x_max)
        log_sum_exp = np.log(np.sum(exp_x, axis=dim, keepdims=True))
        out = Value(self.data - x_max - log_sum_exp, (self,), 'log_softmax')

        def _backward():
            # Gradient of log_softmax: grad_i = grad_out_i - exp(log_softmax_i) * sum(grad_out_j)
            softmax = np.exp(out.data)
            grad_sum = np.sum(out.grad, axis=dim, keepdims=True)
            self.grad += out.grad - softmax * grad_sum
        
        out._backward = _backward

        return out

    def cross_entropy(self, targets):
        """
        Compute cross entropy loss.
        Args:
            self: logits of shape (B, T, vocab_size) or (B*T, vocab_size)
            targets: target indices of shape (B, T) or (B*T)
        Returns:
            loss: scalar Value
        """
        # Reshape if needed
        if len(self.shape) == 3:
            B, T, C = self.shape
            logits = self.reshape(B*T, C)
            targets = targets.reshape(B*T)
        else:
            logits = self

        # Compute log probabilities
        log_probs = logits.log_softmax(dim=-1)
        
        # Gather correct token log probs using numpy advanced indexing
        correct_log_probs = log_probs.data[np.arange(len(targets.data)), targets.data]
        
        # Mean negative log probability
        loss = Value(-np.mean(correct_log_probs), (log_probs,), 'cross_entropy')

        def _backward():
            # Gradient is softmax - one_hot
            softmax = np.exp(log_probs.data)
            softmax[np.arange(len(targets.data)), targets.data] -= 1
            log_probs.grad += softmax * loss.grad / len(targets.data)
        loss._backward = _backward

        return loss

    def layer_norm(self, gamma=None, beta=None, eps=1e-5):
        """Layer normalization with optional affine transform.
        Args:
            gamma: Scale parameter (learnable)
            beta: Shift parameter (learnable)
            eps: Small constant for numerical stability
        """
        # Calculate mean and variance along last dimension
        mean = np.mean(self.data, axis=-1, keepdims=True)
        var = np.var(self.data, axis=-1, keepdims=True)

        #Normalize and apply affine transform if provided
        x_norm = (self.data - mean) / np.sqrt(var + eps)
        
        if gamma is not None and beta is not None:
            x_norm = gamma.data * x_norm + beta.data

        out = Value(x_norm, (self, gamma, beta) if gamma is not None else (self,), 'layer_norm')

        def _backward():
            N = self.shape[-1]
            x_centered = self.data - mean
            std_inv = 1 / np.sqrt(var + eps)

            if gamma is not None:
                #Gradient through affine transform
                grad_in = out.grad * gamma.data
                gamma.grad += np.sum(out.grad * x_norm, axis=0)
                if beta is not None:
                    beta.grad += np.sum(out.grad, axis=0)
            else:
                grad_in = out.grad
                
            # Gradients from standardization
            grad_var = np.sum(grad_in * x_centered * -0.5 * std_inv**3, axis=-1, keepdims=True)
            grad_mean = np.sum(grad_in * -std_inv, axis=-1, keepdims=True)
            
            # Combine all terms
            self.grad += (grad_in * std_inv + 
                        2 * grad_var * x_centered / N +
                        grad_mean / N)
        out._backward = _backward
        
        return out

    def embedding(self, indices, embedding_matrix):
        """
        Purpose: Convert token IDs into vectors (like a lookup table)
        
        Example:
        vocab_size = 1000
        embed_dim = 16
        embedding_matrix = Value(np.random.randn(vocab_size, embed_dim))
        indices = [1, 4, 2]  # Three token IDs
        vectors = embedding(indices, embedding_matrix)  # Shape: (3, embed_dim)
        """
        # indices=[1,4,2] will select rows 1,4,2 from embedding_matrix
        out_data = embedding_matrix.data[indices]
        out = Value(out_data, (embedding_matrix,), 'embedding')
        
        def _backward():
            # During backprop, we need to accumulate gradients for each used embedding
            # np.add.at allows adding to the same index multiple times
            np.add.at(embedding_matrix.grad, indices.data, out.grad)
       
        out._backward = _backward
        
        return out
    
    def dropout(self, p=0.5, training=True):
        """
        Purpose: Randomly zero out elements to prevent overfitting
        
        Example:
        x = Value(np.random.randn(5, 10))  # Some activations
        # During training: randomly zero 50% of elements
        x_dropped = x.dropout(p=0.5, training=True)
        # During inference: no dropout
        x_eval = x.dropout(p=0.5, training=False)
        """
        if training:
            # Create random mask: 1s with prob (1-p), 0s with prob p
            mask = np.random.binomial(1, 1-p, size=self.shape)
            # Scale by 1/(1-p) to maintain expected value
            mask = mask / (1-p)
            out = Value(self.data * mask, (self,), 'dropout')
            
            def _backward():
                # Backprop through the same mask
                self.grad += out.grad * mask
            out._backward = _backward
        else:
            # During evaluation, just pass through
            out = self
        
        return out
    
    def causal_mask(self, size):
        """
        Purpose: Create mask for attention to prevent looking at future tokens
        
        Example:
        size = 4
        mask = causal_mask(size)
        # Returns:
        # [[0, 1, 1, 1],  # First token can only look at itself
        #  [0, 0, 1, 1],  # Second token can look at first and itself
        #  [0, 0, 0, 1],  # Third token can look at first, second, and itself
        #  [0, 0, 0, 0]]  # Fourth token can look at all previous tokens
        # Where 1 means "mask out" (can't look at) and 0 means "can attend to"
        """
        # np.triu creates upper triangular matrix
        # k=1 means start one above diagonal
        mask = np.triu(np.ones((size, size)), k=1).astype(bool)
        return mask

    # Define sigmoid and swish to build the SwiGLU activation function
    def sigmoid(self):
        out = Value(1 / (1 + np.exp(-self.data)), (self, ), 'sigmoid')

        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward
        
        return out
    
    def swish(self):
        sig = self.sigmoid()
        out = Value(self.data* sig.data, (self, ), 'swish')

        def _backward():
            self.grad += (sig.data + self.data * sig.data * (1 - sig.data)) * out.grad
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
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()