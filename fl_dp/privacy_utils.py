import torch
import numpy as np
from typing import List, Tuple

class PrivacyEngine:
    def __init__(self, 
                 noise_multiplier: float = 1.0,
                 max_grad_norm: float = 1.0,
                 noise_scale: float = 0.1):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.noise_scale = noise_scale

    def add_noise_to_gradients(self, model: torch.nn.Module) -> None:
        """Add calibrated noise to gradients"""
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.noise_multiplier * self.max_grad_norm
                    param.grad += noise

    def clip_gradients(self, model: torch.nn.Module) -> None:
        """Clip gradients to max norm"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform mixup augmentation on the input data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

def add_sample_noise(logits: torch.Tensor, temperature: float = 1.0, noise_scale: float = 0.1) -> torch.Tensor:
    """Add noise to logits during sampling"""
    noise = torch.randn_like(logits) * noise_scale
    return (logits + noise) / temperature

class SecureAggregation:
    """Simple secure aggregation implementation"""
    def __init__(self, noise_scale: float = 0.01):
        self.noise_scale = noise_scale

    def aggregate_models(self, model_states: List[dict]) -> dict:
        """Aggregate models with differential privacy"""
        aggregated_state = {}
        for key in model_states[0].keys():
            # Stack parameters
            stacked_params = torch.stack([state[key].float() for state in model_states])
            
            # Add noise for privacy
            noise = torch.randn_like(stacked_params.mean(dim=0)) * self.noise_scale
            
            # Average and add noise
            aggregated_state[key] = stacked_params.mean(dim=0) + noise
        
        return aggregated_state