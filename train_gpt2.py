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
	n_layer: int = 6
 
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
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Bundle q, k, v, and output projection parameters together
        # First 3*n_embd rows are qkv; last n_embd rows are the output projection
        self.attn = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.n_head = config.n_head
        self.d_head = config.n_embd // config.n_head

    def forward(self, x):
        B, T, C = x.size()
        # use the first 3*C rows of the weight/bias for qkv projection
        W_qkv = self.attn.weight[: 3 * C, :]
        b_qkv = self.attn.bias[: 3 * C] if self.attn.bias is not None else None
        qkv = F.linear(x, W_qkv, b_qkv).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(2)
        # compute attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        attn = attn.softmax(dim=-1)
        # apply attention
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        # apply the output projection using the last C rows of the bundled weight/bias
        W_o = self.attn.weight[3 * C : 4 * C, :]
        b_o = self.attn.bias[3 * C : 4 * C] if self.attn.bias is not None else None
        return F.linear(out, W_o, b_o)

