import os
import math
import time
import inspect
from dataclasses import dataclass
from centigrad.engine import Value
import centigrad.NN as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

