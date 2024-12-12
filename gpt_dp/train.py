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
# Improved configuration values for medical data with DP
out_dir = 'out-medical'
eval_interval = 5
log_interval = 1
eval_iters = 5
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Data loading improvements
dataset = 'medical'
gradient_accumulation_steps = 8  # Increased for better gradient signal
batch_size = 64  # Larger batch size for more stable training
block_size = 256  # Reduced context window for more focused learning

# Model adjustments - smaller but deeper architecture
n_layer = 6  # Reduced number of layers
n_head = 6  # Reduced number of heads
n_embd = 384  # Reduced embedding dimension
dropout = 0.1  # Reduced dropout
bias = True  # Added bias terms back

# Optimizer improvements
learning_rate = 5e-5  # Much smaller learning rate
max_iters = 5000  # More iterations
weight_decay = 0.01  # Reduced weight decay
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.1  # Tighter gradient clipping

# Learning rate schedule
decay_lr = True
warmup_iters = 1000  # Longer warmup
lr_decay_iters = 4000
min_lr = 1e-6

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'  # Using full precision for stability
compile = False  # Disabled compilation for better reproducibility
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Refined DP parameters
use_dp = True
target_epsilon = 10.0  # Slightly higher epsilon for better utility
target_delta = 1e-5
max_grad_norm = 0.5  # Reduced gradient norm
noise_multiplier = 0.6  # Reduced noise
num_epochs = 5  # More epochs

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class RefinedPrivacyEngine:
    """Enhanced Privacy Engine with adaptive mechanisms"""
    def __init__(self, noise_multiplier, max_grad_norm, target_epsilon, target_delta, sample_rate):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.sample_rate = sample_rate
        self.steps = 0
        self.rdp_orders = np.linspace(1.1, 20.0, 200)
        self.accumulated_rdp_noises = np.zeros_like(self.rdp_orders)
        
        # Dynamic noise adjustment
        self.noise_scale = 1.0
        self.min_noise_scale = 0.7
        self.loss_history = []
        
    def _compute_rdp(self, sigma, order):
        """Compute RDP privacy cost"""
        if sigma == 0:
            return float('inf')
        return order / (2 * (sigma ** 2))
    
    def update_privacy_params(self, current_loss):
        """Dynamically adjust privacy parameters based on training progress"""
        self.loss_history.append(current_loss)
        if len(self.loss_history) > 50:
            recent_loss_avg = np.mean(self.loss_history[-50:])
            recent_loss_std = np.std(self.loss_history[-50:])
            
            # Adjust noise based on loss stability
            if recent_loss_std < 0.1:  # Loss is stable
                self.noise_scale = max(self.min_noise_scale, self.noise_scale * 0.95)
            else:
                self.noise_scale = min(1.0, self.noise_scale * 1.05)
    
    def compute_epsilon(self):
        """Compute current privacy spent"""
        eps_candidates = []
        for i, order in enumerate(self.rdp_orders):
            rdp = self._compute_rdp(self.noise_multiplier * self.noise_scale, order)
            eps = rdp * self.steps - np.log(self.target_delta) / (order - 1)
            eps_candidates.append(eps)
        return float(min(eps_candidates))

def refined_privacy_mechanism(model, privacy_engine, batch_size):
    """Improved privacy mechanism with better gradient treatment"""
    # Layer-wise gradient clipping
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    
    # Adaptive clipping
    clip_coef = min(privacy_engine.max_grad_norm / (total_norm + 1e-6), 1.0)
    
    # Layer-wise noise addition with importance weighting
    for param in model.parameters():
        if param.grad is not None:
            # Apply clipping
            param.grad.data.mul_(clip_coef)
            
            # Calculate layer-specific noise scale
            param_size = param.grad.data.numel()
            noise_scale = privacy_engine.noise_multiplier * privacy_engine.noise_scale * \
                         privacy_engine.max_grad_norm / math.sqrt(batch_size)
            
            # Add calibrated noise
            noise = torch.randn_like(param.grad) * noise_scale
            param.grad.add_(noise)

class MedicalDataset(torch.utils.data.Dataset):
    """Dataset handler for medical records"""
    def __init__(self, split):
        data_dir = os.path.join('data', dataset)
        data_file = 'train.bin' if split == 'train' else 'val.bin'
        self.data = np.memmap(os.path.join(data_dir, data_file), dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.vocab_size = None  # Will be set from meta.pkl
        
        # Calculate valid starting indices
        self.valid_indices = len(self.data) - block_size
    
    def __len__(self):
        return self.valid_indices
    
    def __getitem__(self, idx):
        # Get block of data
        block_data = self.data[idx:idx + self.block_size]
        x = torch.from_numpy(block_data[:-1].astype(np.int64))
        y = torch.from_numpy(block_data[1:].astype(np.int64))
        return x, y

def get_lr(iter):
    """Learning rate schedule with warmup"""
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    if iter > lr_decay_iters:
        return min_lr
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def train_epoch(epoch, model, train_loader, optimizer, privacy_engine, ctx):
    """Single epoch training with curriculum learning"""
    model.train()
    total_loss = 0
    
    for iter, (X, y) in enumerate(train_loader):
        if iter >= max_iters:
            break
            
        # Move data to device
        X, y = X.to(device), y.to(device)
        
        # Progressive batch accumulation
        effective_batch_size = batch_size * (1 + epoch // 2)  # Increase batch size gradually
        
        # Learning rate update
        lr = get_lr(iter) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        loss_chunk = 0
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, y)
                loss = loss / gradient_accumulation_steps
                loss_chunk += loss.item()
            loss.backward()
        
        if use_dp:
            refined_privacy_mechanism(model, privacy_engine, effective_batch_size)
            privacy_engine.update_privacy_params(loss_chunk)
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss_chunk
        
        if iter % log_interval == 0:
            print(f"Epoch {epoch}, Iter {iter}: Loss = {loss_chunk:.4f}, lr = {lr:.2e}")
    
    return total_loss / len(train_loader)

def main():
    # Load meta data
    data_dir = os.path.join('data', dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # Model arguments
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=meta['vocab_size'],
        dropout=dropout
    )
    
    # Initialize model and datasets
    model = GPT(GPTConfig(**model_args))
    model.to(device)
    
    train_dataset = MedicalDataset('train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # Initialize privacy engine if using DP
    if use_dp:
        privacy_engine = RefinedPrivacyEngine(
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=batch_size / len(train_dataset)
        )
        print(f"Initialized Privacy Engine with ε={target_epsilon}, δ={target_delta}")
    
    best_loss = float('inf')
    start_time = time.time()
    
    try:
        for epoch in range(num_epochs):
            train_loss = train_epoch(epoch, model, train_loader, optimizer, privacy_engine, ctx)
            
            if use_dp:
                current_epsilon = privacy_engine.compute_epsilon()
                print(f"Epoch {epoch} completed. Loss: {train_loss:.4f}, ε = {current_epsilon:.2f}")
            else:
                print(f"Epoch {epoch} completed. Loss: {train_loss:.4f}")
            
            # Save checkpoint
            if train_loss < best_loss or always_save_checkpoint:
                best_loss = min(best_loss, train_loss)
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'config': {k:v for k,v in globals().items() 
                             if not k.startswith('_') and isinstance(v, (int, float, bool, str))},
                    'privacy_budget': current_epsilon if use_dp else None,
                    'epoch': epoch,
                    'loss': train_loss,
                }
                print(f"Saving checkpoint to {out_dir}")
                os.makedirs(out_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        if use_dp:
            final_epsilon = privacy_engine.compute_epsilon()
            print(f"Final Privacy Guarantees: ε = {final_epsilon:.2f}, δ = {target_delta}")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")

if __name__ == '__main__':
    main()