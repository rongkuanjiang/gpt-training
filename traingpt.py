import torch

with open('input.txt', 'r', encoding='utf-8') as f:
	data = f.read()

print("length of database:", len(data))
print(data[0:1000])

chars = sorted(list(set(data)))
vocab_size = len(chars)
print(vocab_size)
print("".join(chars))

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(data), dtype=torch.long)
print(data.shape, data.dtype)
print(data[0:1000])
print(torch.cuda.is_available())

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

