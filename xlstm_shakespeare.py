import torch
import torch.nn as nn
from torch.nn import functional as F

from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from omegaconf import OmegaConf
from dacite import from_dict

import time

import numpy as np

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

def load_data():
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()    
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size, decode

def get_batch(split, train_data, val_data, config):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.dataset.context_length, (config.training.batch_size,))
    x = torch.stack([data[i:i+config.dataset.context_length] for i in ix])
    y = torch.stack([data[i+1:i+config.dataset.context_length+1] for i in ix])
    x, y = x.to(config.training.device), y.to(config.training.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.training.eval_iters)
        for k in range(config.training.eval_iters):
            X, Y = get_batch(split, train_data, val_data, config)
            logits = model(X)
            loss = nn.functional.cross_entropy(
                logits.view(-1, config.model.vocab_size),
                Y.view(-1),
                ignore_index=-1,
            )
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def generate(model, config, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -config.dataset.context_length:]
        # get the predictions
        logits = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

yaml_cfg ="""
training:
  batch_size: 16
  lr: 0.001
  eval_interval: 100
  num_steps: 1000
  device: cuda
  eval_iters: 20
  enable_mixed_precision: true
  amp_precision: bfloat16
  weight_precision: float32

model:
  num_blocks: 2
  embedding_dim: 32
  mlstm_block:
    mlstm:
      num_heads: 4
  slstm_block:
    slstm:
      num_heads: 4
  slstm_at: [1]
  dropout: 0.2
  context_length: ${dataset.context_length}

dataset:
  name: tinyshakespeare
  context_length: 8
"""
#train_data, val_data, vocab_size, decode = load_data()

torch.manual_seed(42)

config = OmegaConf.create(yaml_cfg)
OmegaConf.resolve(config)
train_data, val_data, vocab_size, decode = load_data()
config.model.vocab_size = vocab_size

model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(config.model))).to(device=config.training.device)
model.reset_parameters()
model = model.to(dtype=torch_dtype_map[config.training.weight_precision])
num_params = sum(p.numel() for p in model.parameters())
print(num_params/1e6, 'M parameters')

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
gpu_utilization = []

start_time = time.time()

for step in range(config.training.num_steps):

    # sample a batch of data
    xb, yb = get_batch('train', train_data, val_data, config)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=config.training.device,
        dtype=torch_dtype_map[config.training.amp_precision],
        enabled=config.training.enable_mixed_precision,
    ):
        logits = model(xb)
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.model.vocab_size),
            yb.view(-1),
            ignore_index=-1,
        )
        loss.backward()
        optimizer.step()
        # every once in a while evaluate the loss on train and val sets
        if step % config.training.eval_interval == 0 or iter == config.training.num_steps - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            gpu_utilization.append(torch.cuda.utilization())

end_time = time.time()
print("=== Finished (XLSTM) ===")
print(f"{config.training.num_steps} steps, {config.training.batch_size} batch size, final losses: {losses['train']:.4f} train, {losses['val']:.4f} val")
print(f"{num_params:,} parameters, {config.dataset.context_length} context length")
print(f"{sum(gpu_utilization)/len(gpu_utilization):.1f}% average GPU utilization")
print("Wall clock time elapsed: %.2f seconds" % (end_time-start_time))

# Generate an example text
context = torch.zeros((1, 1), dtype=torch.long, device=config.training.device)
print(decode(generate(model, config, context, max_new_tokens=500)[0].tolist()))