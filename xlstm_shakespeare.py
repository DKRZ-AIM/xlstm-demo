import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from omegaconf import OmegaConf
from dacite import from_dict

import numpy as np

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class TinyShakespeareDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()    
        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        self.encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        self.decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string    
        self.data = torch.tensor(self.encode(text), dtype=torch.long).to(self.config.training.device)
        n = int(0.9*len(self.data)) # first 90% will be train, rest val
        if train:
            self.data = self.data[:n]
        else:
            self.data = self.data[n:]
        self.datalen = len(self.data)

    def __len__(self):
        return self.datalen

    def __getitem__(self, i):
        #ix = torch.randint(self.datalen - config.dataset.context_length, (config.training.batch_size,))
        x = self.data[i:i+self.config.dataset.context_length]
        y = self.data[i+1:i+self.config.dataset.context_length+1]
        #x, y = x.to(config.training.device), y.to(config.training.device)
        return x, y

def create_data_loader(config, train):
    dataset = TinyShakespeareDataset(config, train)
    loader = DataLoader(dataset, batch_size=config.training.batch_size)
    return loader

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

def train_loop(dataloader, model, loss_fn, optimizer, config):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        #loss = loss_fn(pred, y)
        loss = nn.functional.cross_entropy(
            pred.view(-1, config.model.vocab_size),
            y.view(-1),
            ignore_index=-1,
        )
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * config.training.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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
                logits.view(-1, config.dataset.vocab_size),
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
  epochs: 10
  batch_size: 64
  lr: 0.001
  eval_interval: 100
  num_steps: 5000
  device: cuda
  eval_iters: 20
  enable_mixed_precision: true
  amp_precision: bfloat16
  weight_precision: float32

model:
  num_blocks: 2
  embedding_dim: 64
  mlstm_block:
    mlstm:
      num_heads: 1
  slstm_block:
    slstm:
      num_heads: 1
  slstm_at: [1]
  context_length: ${dataset.context_length}

dataset:
  name: tinyshakespeare
  context_length: 16
"""
#train_data, val_data, vocab_size, decode = load_data()


config = OmegaConf.create(yaml_cfg)
OmegaConf.resolve(config)

train_dataloader = create_data_loader(config, train=True)
test_dataloader =  create_data_loader(config, train=False)

config.model.vocab_size = train_dataloader.dataset.vocab_size

model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(config.model))).to(device=config.training.device)
model.reset_parameters()
model = model.to(dtype=torch_dtype_map[config.training.weight_precision])
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
print("len(model.parameters()): ", len(list(model.parameters())))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)

for t in range(config.training.epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, config)
    test_loop(test_dataloader, model, loss_fn)
    print("Done!")

"""
for batch, (X, y) in enumerate(train_loader):

    # sample a batch of data
    #xb, yb = get_batch('train', train_data, val_data, config)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=config.training.device,
        dtype=torch_dtype_map[config.training.amp_precision],
        enabled=config.training.enable_mixed_precision,
    ):
        logits = model(X)
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.dataset.vocab_size),
            y.view(-1),
            ignore_index=-1,
        )
        loss.backward()
        optimizer.step()
        # every once in a while evaluate the loss on train and val sets
        if batch % config.training.eval_interval == 0 or iter == config.training.num_steps - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            print(f"step {batch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
"""

# Generate an example text
context = torch.zeros((1, 1), dtype=torch.long, device=config.training.device)
print(test_dataloader.dataset.decode(generate(model, config, context, max_new_tokens=500)[0].tolist()))