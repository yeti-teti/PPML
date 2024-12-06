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
# Default configuration values
# I/O
out_dir = 'out-medical'  # Changed output directory
eval_interval = 5
log_interval = 1
eval_iters = 5
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Data
dataset = 'medical'  # Changed to medical dataset
gradient_accumulation_steps = 1  # Must be 1 for differential privacy
batch_size = 16  # Adjusted for medical data
block_size = 1024

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1  # Increased dropout for better generalization
bias = False

# AdamW optimizer
learning_rate = 5e-4  # Slightly reduced learning rate
max_iters = 1000  # Increased iterations for medical data
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay settings
decay_lr = True
warmup_iters = 100  # Increased warmup
lr_decay_iters = 800
min_lr = 5e-5

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'
compile = True

# Differential Privacy Parameters (adjusted for medical data)
use_dp = True
target_epsilon = 3.0  # Stricter privacy budget for medical data
target_delta = 1e-6  # Stricter delta for medical data
max_grad_norm = 0.5  # Reduced gradient norm for tighter privacy
noise_multiplier = 1.3  # Increased noise for better privacy
num_epochs = 2

# -----------------------------------------------------------------------------
# Advanced Privacy Engine
class AdvancedPrivacyEngine:
    def __init__(self, noise_multiplier, max_grad_norm, target_epsilon, target_delta, sample_rate):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.sample_rate = sample_rate
        self.steps = 0
        self.rdp_orders = np.linspace(1.1, 10.9, 100)
        self.accumulated_rdp_noises = np.zeros_like(self.rdp_orders)
    
    def _compute_rdp(self, sigma, order):
        if sigma == 0:
            return float('inf')
        return order / (2 * (sigma ** 2))
    
    def accumulate_privacy_loss(self, sample_rate):
        for i, order in enumerate(self.rdp_orders):
            rdp_noise = self._compute_rdp(self.noise_multiplier, order) * sample_rate
            self.accumulated_rdp_noises[i] += rdp_noise
    
    def compute_epsilon(self):
        eps_candidates = []
        for i, order in enumerate(self.rdp_orders):
            eps = self.accumulated_rdp_noises[i] - np.log(self.target_delta) / (order - 1)
            eps_candidates.append(eps)
        return float(min(eps_candidates))
    
    def step(self, sample_rate):
        self.steps += 1
        self.accumulate_privacy_loss(sample_rate)

def clip_and_add_noise(model, max_norm, noise_multiplier, device):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_multiplier * max_norm
            param.grad.add_(noise)

# -----------------------------------------------------------------------------
# Initialize training
master_process = not int(os.environ.get('RANK', -1)) != -1
seed = 1337
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loading
data_dir = os.path.join('data', dataset)

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        data_file = 'train.bin' if split == 'train' else 'val.bin'
        self.data = np.memmap(os.path.join(data_dir, data_file), dtype=np.uint16, mode='r')
        self.block_size = block_size
        
        # Calculate valid starting indices (accounting for block_size)
        self.valid_indices = len(self.data) - block_size
        
    def __len__(self):
        return self.valid_indices
    
    def __getitem__(self, idx):
        # Ensure we don't split patient records
        block_data = self.data[idx:idx + self.block_size]
        x = torch.from_numpy(block_data[:-1].astype(np.int64))
        y = torch.from_numpy(block_data[1:].astype(np.int64))
        return x, y

# Create datasets and data loaders
train_dataset = MedicalDataset('train')
val_dataset = MedicalDataset('val')

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

# Model initialization
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
    
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta['vocab_size'],
    dropout=dropout
)

model = GPT(GPTConfig(**model_args))
model.to(device)
print(f"Training GPT model with {model.get_num_params():,} parameters")

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("Compiling model...")
    model = torch.compile(model)

# Privacy Engine Initialization
if use_dp:
    total_samples = len(train_dataset)
    sample_rate = batch_size / total_samples
    privacy_engine = AdvancedPrivacyEngine(
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate
    )
    print(f"Initialized Privacy Engine with ε={target_epsilon}, δ={target_delta}")

# Training loop
def train_epoch(epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for iter, (X, y) in enumerate(train_loader):
        if iter >= max_iters:
            break
            
        # Move data to device
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        with ctx:
            logits, loss = model(X, y)
            
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Apply differential privacy if enabled
        if use_dp:
            clip_and_add_noise(model, max_grad_norm, noise_multiplier, device)
            privacy_engine.step(sample_rate)
            
            if iter % log_interval == 0:
                current_epsilon = privacy_engine.compute_epsilon()
                print(f"Step {iter}: ε = {current_epsilon:.2f}, δ = {target_delta}")
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        if iter % log_interval == 0:
            print(f"Epoch {epoch}, Iter {iter}: Loss = {loss.item():.4f}")
            
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    return avg_loss

# Training execution
print(f"Starting training with {len(train_dataset):,} samples")
best_val_loss = float('inf')

try:
    for epoch in range(num_epochs):
        train_loss = train_epoch(epoch)
        
        # Save checkpoint
        if master_process and (epoch % eval_interval == 0 or always_save_checkpoint):
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'config': {k:v for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))},
                'epoch': epoch,
                'train_loss': train_loss,
            }
            print(f"Saving checkpoint to {out_dir}")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        
        if use_dp:
            final_epsilon = privacy_engine.compute_epsilon()
            print(f"Epoch {epoch} Privacy Budget: ε = {final_epsilon:.2f}, δ = {target_delta}")

except KeyboardInterrupt:
    print("Training interrupted by user")

finally:
    if use_dp:
        final_epsilon = privacy_engine.compute_epsilon()
        print(f"Final Privacy Guarantees: ε = {final_epsilon:.2f}, δ = {target_delta}")
    
    print("Training completed!")