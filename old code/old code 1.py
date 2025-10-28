
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)
    
# batch_size = 64
# block_size = 256
# max_iters = 5000
# eval_interval = 300
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2

# @dataclass
# class GPTConfig:
#     vocab_size: int = 50257
#     block_size: int = 1024
#     n_embd: int = 768
#     n_head: int = 12
#     n_layer: int = 12
 
# class GPT(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.transformer = nn.ModuleDict(
#             dict(
#                 wte=nn.Embedding(config.vocab_size, config.n_embd),
#                 wpe=nn.Embedding(config.block_size, config.n_embd),
#                 h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
#                 ln_f=nn.LayerNorm(config.n_embd),
#             )
#         )
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
#         # weight tying
#         self.lm_head.weight = self.transformer.wte.weight

#     def load_pretrained(self, model_type):
#         """Loads pretrained GPT-2 model weights from huggingface"""
#         assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
#         from transformers import GPT2LMHeadModel
#         print("loading weights from pretrained gpt: %s" % model_type)

#         # n_layer, n_head and n_embd are determined from model_type
#         config_args = {
#             'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
#             'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
#             'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
#             'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
#         }[model_type]
#         config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
#         config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
#         # create a from-scratch initialized minGPT model
#         config = GPTConfig(**config_args)
#         model = GPT(config)
#         sd = model.state_dict()
#         sd_keys = sd.keys()
#         sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

#         # init a huggingface/transformers model
#         model_hf = GPT2LMHeadModel.from_pretrained(model_type)
#         sd_hf = model_hf.state_dict()

#         # copy while ensuring all of the parameters are aligned and match in names and shapes
#         sd_keys_hf = sd_hf.keys()
#         sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
#         sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
#         transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
#         # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
#         # this means that we have to transpose these weights when we import them
#         assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
#         for k in sd_keys_hf:
#             if any(k.endswith(w) for w in transposed):
#                 # special treatment for the Conv1D weights we need to transpose
#                 assert sd_hf[k].shape[::-1] == sd[k].shape
#                 with torch.no_grad():
#                     sd[k].copy_(sd_hf[k].t())
#             else:
#                 # vanilla copy over the other parameters
#                 assert sd_hf[k].shape == sd[k].shape
#                 with torch.no_grad():
#                     sd[k].copy_(sd_hf[k])

#         return model


# class Block(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(config.n_embd)
#         self.attn = MultiHeadAttention(config)
#         self.ln2 = nn.LayerNorm(config.n_embd)
#         self.mlp = MLP(config)
        
#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x

# class MLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.up = nn.Linear(config.n_embd, 4 * config.n_embd)
#         self.down = nn.Linear(config.n_embd * 4, config.n_embd)
#         self.gelu = nn.GELU(approximate='tanh')
#         self.dropout = nn.Dropout(0.1)
    
#     def forward(self, x):
#         x = self.up(x)
#         x = self.gelu(x)
#         x = self.down(x)
#         x = self.dropout(x)
#         return x
# class MultiHeadAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0
#         self.w_qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
#         self.w_o = nn.Linear(config.n_embd, config.n_embd)
#         self.n_head = config.n_head
#         self.d_head = config.n_embd // config.n_head

#     def forward(self, x):
#         B, T, C = x.size()
        
#         qkv = self.w_qkv(x)
#         # b_qkv = self.attn.bias[: 3 * C] if self.attn.bias is not None else None
#         q, k, v = qkv.split(x.size(-1), dim=2)
#         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
#         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
#         #this just does EVERYTHIN (att matrix, multiplyin, softmax)
#         y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.w_o(y)
#         return y

# #GOURT: YO




