# client.py
import flwr as fl
import torch
from torch.utils.data import DataLoader

from model import GPT, GPTConfig 
from train import (
    ShakespeareDataset,
    GPTConfig,  # If needed
    # Include any other necessary imports or helper functions
)
import os
import numpy as np

# Privacy Engine Implementation (Assuming you have a custom PrivacyEngine)
# If you're using a library like Opacus, you can integrate it accordingly.
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

    def clip_gradients(self, model, max_norm):
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

    def add_noise(self, model, noise_multiplier, max_grad_norm, device):
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_multiplier * max_grad_norm
                param.grad += noise.to(device)

# Define the Flower client
class GPTClient(fl.client.NumPyClient):
    def __init__(self, model: GPT, train_loader: DataLoader, val_loader: DataLoader, optimizer, config, privacy_engine=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.privacy_engine = privacy_engine

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, param in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(self.config['num_epochs']):
            for batch_idx, (X, Y) in enumerate(self.train_loader):
                X, Y = X.to(self.model.device).long(), Y.to(self.model.device).long()
                self.optimizer.zero_grad()
                logits, loss = self.model(X, Y)
                loss.backward()
                if self.privacy_engine:
                    # Apply differential privacy steps
                    self.privacy_engine.clip_gradients(self.model, self.config['max_grad_norm'])
                    self.privacy_engine.add_noise(self.model, self.config['noise_multiplier'], self.config['max_grad_norm'], self.model.device)
                    self.privacy_engine.step()
                if self.config['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for X, Y in self.val_loader:
                X, Y = X.to(self.model.device).long(), Y.to(self.model.device).long()
                logits, loss = self.model(X, Y)
                total_loss += loss.item() * X.size(0)
                total_samples += X.size(0)
        average_loss = total_loss / total_samples
        return float(average_loss), total_samples, {"loss": average_loss}

def main():
    import math  # Needed for PrivacyEngine

    # Configuration (same as in train.py or adjust as needed)
    config = {
        'out_dir': 'out',
        'batch_size': 12,
        'block_size': 1024,
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'dropout': 0.0,
        'bias': False,
        'learning_rate': 6e-4,
        'weight_decay': 1e-1,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        'use_dp': True,
        'max_grad_norm': 1.0,
        'noise_multiplier': 1.1,
        'num_epochs': 1,
        'dataset': 'shakespeare',
        'target_delta': 1e-5,  # Add target_delta for PrivacyEngine
    }

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model
    model_args = dict(
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        block_size=config['block_size'],
        bias=config['bias'],
        vocab_size=50304,  # Adjust as needed
        dropout=config['dropout']
    )
    model = GPT(GPTConfig(**model_args)).to(device)

    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config['weight_decay'],
        learning_rate=config['learning_rate'],
        betas=(config['beta1'], config['beta2']),
        device_type='cuda' if device.startswith('cuda') else 'cpu'
    )

    # Load data
    train_dataset = ShakespeareDataset('train')
    val_dataset = ShakespeareDataset('val')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # Initialize Privacy Engine if using DP
    privacy_engine = None
    if config['use_dp']:
        sample_rate = (config['batch_size']) / len(train_dataset)
        privacy_engine = PrivacyEngine(
            noise_multiplier=config['noise_multiplier'],
            max_grad_norm=config['max_grad_norm'],
            target_delta=config['target_delta'],
            sample_rate=sample_rate
        )
        # Ensure PrivacyEngine methods are accessible or adjust accordingly

    # Initialize Flower client
    client = GPTClient(model, train_loader, val_loader, optimizer, config, privacy_engine)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
