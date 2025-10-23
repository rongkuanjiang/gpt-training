import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

from nn import MLP

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

@dataclass
class GPTConfig:
	vocab_size: int = 256
	block_size: int = 65
	n_embd: int = 6
	n_head: int = 6
	n_layer	: int = 6
 
class GPT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.transformer = nn.ModuleDict(dict(
  			wte = nn.Embedding(config.vocab_size, config.n_embd),
  			wpe = nn.Embedding(config.block_size, config.n_embd),
			h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
  			ln_f = nn.LayerNorm(config.n_embd)
  		))
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()