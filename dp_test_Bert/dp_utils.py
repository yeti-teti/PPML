import math
import numpy as np
import torch
from collections import defaultdict

class RDPAccountant:
    """Privacy accounting using Renyi Differential Privacy"""
    def __init__(self, noise_multiplier, target_epsilon, target_delta, sample_rate):
        self.noise_multiplier = noise_multiplier
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.sample_rate = sample_rate
        self.steps = 0
        self.rdp_orders = np.linspace(1.1, 20.0, 200)
        self.rdp_alpha_schedule = defaultdict(float)
    
    def compute_rdp(self, alpha):
        """Compute RDP at order alpha"""
        return alpha / (2 * (self.noise_multiplier ** 2))
    
    def step(self):
        """Account for privacy cost of one step"""
        self.steps += 1
        for alpha in self.rdp_orders:
            rdp_step = self.compute_rdp(alpha) * self.sample_rate
            self.rdp_alpha_schedule[alpha] += rdp_step
    
    def get_epsilon(self):
        """Convert RDP to (ε, δ)-DP"""
        eps_candidates = []
        for alpha in self.rdp_orders:
            eps = self.rdp_alpha_schedule[alpha] - math.log(self.target_delta) / (alpha - 1)
            eps_candidates.append(eps)
        return float(min(eps_candidates))

class PrivacyEngine:
    """Privacy engine for gradient processing"""
    def __init__(self, model, batch_size, max_grad_norm, noise_multiplier):
        self.model = model
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
    
    def clip_gradients(self):
        """Clip gradients by global norm"""
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters]))
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        clip_coef = torch.min(torch.tensor(1.0), clip_coef)
        
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
        
        return total_norm
    
    def add_noise(self):
        """Add calibrated noise to gradients"""
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        noise_scale = self.noise_multiplier * self.max_grad_norm / self.batch_size
        
        for p in parameters:
            noise = torch.randn_like(p.grad) * noise_scale
            p.grad.add_(noise)

def compute_snr(grad_norm, noise_multiplier, max_grad_norm, batch_size):
    """Compute gradient signal-to-noise ratio"""
    noise_norm = noise_multiplier * max_grad_norm / math.sqrt(batch_size)
    return grad_norm / noise_norm