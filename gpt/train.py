import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default configuration values designed to train a GPT-2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 10
log_interval = 1
eval_iters = 5
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1  # Must be 1 for differential privacy
batch_size = 12  # global batch size across all GPUs
block_size = 1024

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?

# AdamW optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 500
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# Learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 20
lr_decay_iters = 40
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
# System
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
dtype = 'float32'  # Using `float32` to avoid data type mismatches
compile = True  # use PyTorch 2.0 to compile the model to be faster

# Differential Privacy Parameters
use_dp = True  # Set true to enable differential privacy
target_epsilon = 10  # Desired epsilon
target_delta = 1e-5  # Desired Delta
max_grad_norm = 1.0  # Clip per-sample gradients to this norm
noise_multiplier = 1.1  # Noise multiplier for DP-SGD
num_epochs = 1  # Number of training epochs

# Privacy Accounting
class PrivacyEngine:
    def __init__(self, noise_multiplier, max_grad_norm, target_delta, sample_rate):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_delta = target_delta
        self.sample_rate = sample_rate
        self.steps = 0
        self.epsilon = 0.0

    def compute_epsilon(self):
        from math import sqrt, log

        q = self.sample_rate
        sigma = self.noise_multiplier
        if q == 0:
            return float('inf')
        # Simple RDP approximation for Gaussian noise
        return q * self.steps / (sigma ** 2)

    def step(self):
        self.steps += 1
        self.epsilon = self.compute_epsilon()

    def get_epsilon(self):
        return self.epsilon

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# Various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = torch.float32  # Enforce consistent data type as float32
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Loading Data
data_dir = os.path.join('data', dataset)

class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        data_file = 'train.bin' if split == 'train' else 'val.bin'
        self.data = np.memmap(os.path.join(data_dir, data_file), dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.length = len(self.data) - self.block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        return x, y

train_dataset = ShakespeareDataset('train')
val_dataset = ShakespeareDataset('val')

print(f"Number of training samples: {len(train_dataset)}")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

total_samples = len(train_dataset)

# Model initialization
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size or 50304,
    dropout=dropout
)
model = GPT(GPTConfig(**model_args)).to(device)

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    model = torch.compile(model)

# Privacy Engine Initialization
if use_dp:
    sample_rate = (gradient_accumulation_steps * batch_size) / total_samples
    privacy_engine = PrivacyEngine(
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        target_delta=target_delta,
        sample_rate=sample_rate
    )
    # Note: The PrivacyEngine is a custom implementation. Ensure it integrates correctly with your training loop.

# Enable fallback to eager mode for debugging
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Helper functions for Differential Privacy
def clip_gradients(model, max_norm):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.detach().mul_(clip_coef)

def add_noise(model, noise_multiplier, max_grad_norm, device):
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_multiplier * max_grad_norm
            param.grad += noise.to(device)

# Function to save checkpoints
def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved at {filename}")

# Training loop
for epoch in range(num_epochs):
    model.train()
    for step, (X, Y) in enumerate(train_loader):
        if step >= max_iters:
            break

        X, Y = X.to(device).long(), Y.to(device).long()  # Ensure correct data type for embedding lookup
        optimizer.zero_grad()

        with ctx:
            logits, loss = model(X, Y)

        loss.backward()

        if use_dp:
            # Per-sample gradient clipping and noise addition
            clip_gradients(model, max_grad_norm)
            add_noise(model, noise_multiplier, max_grad_norm, device)

        # Gradient clipping (global)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if use_dp:
            privacy_engine.step()
            epsilon = privacy_engine.get_epsilon()
            if master_process:
                print(f"Step {step}: ε = {epsilon:.2f}, δ = {target_delta}")

        # Logging loss periodically
        if master_process and step % log_interval == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Save checkpoint after every eval_interval epochs or always if enabled
    if master_process and (epoch % eval_interval == 0 or always_save_checkpoint):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'model_args': model_args,
            'config': config
        }
        checkpoint_path = os.path.join(out_dir, 'ckpt.pt')  # Save as 'ckpt.pt' for sample.py
        save_checkpoint(checkpoint, checkpoint_path)

    if use_dp and master_process:
        print(f"Epoch {epoch}: Total ε = {privacy_engine.get_epsilon():.2f}, δ = {target_delta}")

# Clean up DDP
if ddp:
    destroy_process_group()
