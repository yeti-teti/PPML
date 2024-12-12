import os
import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
import math

from model import GPTConfig, GPT
from dp_utils import RDPAccountant, PrivacyEngine, compute_snr

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, block_size):
        self.block_size = block_size
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.data = torch.from_numpy(data.astype(np.int64))
        self.total_len = len(self.data) - block_size  # Account for sequence length

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Get block_size + 1 tokens (input + target)
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]  # Input sequence
        y = chunk[1:]   # Target sequence
        return x, y

def create_dataloader(data_path, batch_size, block_size):
    dataset = TextDataset(data_path, block_size)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

def train(
    out_dir='out-medical',
    eval_interval=5,
    log_interval=1,
    eval_iters=5,
    dataset='medical',
    gradient_accumulation_steps=8,
    batch_size=64,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True,
    learning_rate=5e-5,
    max_iters=5000,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.999,
    grad_clip=0.1,
    decay_lr=True,
    warmup_iters=1000,
    lr_decay_iters=4000,
    min_lr=1e-6,
    device='cuda',
    dtype='float32',
    compile=False,
    num_epochs=5,
    # DP parameters
    epsilon=8.0,
    delta=1e-5,
    max_grad_norm=0.5,
    noise_multiplier=0.6
):
    # Set up device
    device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    
    # Load data
    data_dir = os.path.join('data', dataset)
    train_loader = create_dataloader(
        os.path.join(data_dir, 'train.bin'),
        batch_size,
        block_size
    )
    val_loader = create_dataloader(
        os.path.join(data_dir, 'val.bin'),
        batch_size,
        block_size
    )
    
    # Load metadata
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    # Initialize model
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
    
    if compile:
        model = torch.compile(model)
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay,
        learning_rate,
        (beta1, beta2),
        device
    )
    
    # Initialize privacy engines
    privacy_engine = PrivacyEngine(
        model,
        batch_size,
        max_grad_norm,
        noise_multiplier
    )
    
    accountant = RDPAccountant(
        noise_multiplier,
        epsilon,
        delta,
        batch_size / len(train_loader.dataset)
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for it, (X, y) in pbar:
            # Get batch and verify shapes
            X, y = X.to(device), y.to(device)
            
            # Determine learning rate
            if decay_lr:
                lr = get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Forward pass
            with ctx:
                # Forward pass with targets
                logits, loss = model(X, targets=y)
            
            # Backward pass
            loss.backward()
            
            # Apply DP
            grad_norm = privacy_engine.clip_gradients()
            privacy_engine.add_noise()
            accountant.step()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Logging
            total_loss += loss.item()
            if it % log_interval == 0:
                current_eps = accountant.get_epsilon()
                snr = compute_snr(grad_norm, noise_multiplier, max_grad_norm, batch_size)
                pbar.set_description(f"epoch {epoch+1} iter {it}: loss {loss.item():.4f}, ε = {current_eps:.2f}, SNR = {snr:.4f}")
        
        # Evaluation
        if epoch % eval_interval == 0:
            model.eval()
            val_loss = evaluate(model, val_loader, ctx, device)
            print(f"Epoch {epoch+1}: val_loss = {val_loss:.4f}")
            
            # Save checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_args': model_args,
                    'epsilon': accountant.get_epsilon(),
                    'epoch': epoch,
                    'loss': val_loss
                }
                os.makedirs(out_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

def evaluate(model, val_loader, ctx, device):
    losses = []
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            with ctx:
                logits, loss = model(X, targets=y)
        losses.append(loss.item())
    return np.mean(losses)

def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine decay
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if __name__ == '__main__':
    train()