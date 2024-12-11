import os
import torch
from model import GPTConfig, GPT
from privacy_utils import SecureAggregation

class FederatedServer:
    def __init__(self, noise_scale=0.01):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.init_model()
        self.secure_aggregator = SecureAggregation(noise_scale=noise_scale)
        print(f"Server initialized with secure aggregation (device: {self.device})")

    def init_model(self):
        model_args = dict(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=1024,
            bias=False,
            vocab_size=50257,
            dropout=0.1
        )
        model = GPT(GPTConfig(**model_args))
        return model.to(self.device)

    def aggregate_models(self, client_models):
        """Securely aggregate client models"""
        aggregated_state = self.secure_aggregator.aggregate_models(client_models)
        self.model.load_state_dict(aggregated_state)

    def get_model_state(self):
        return self.model.state_dict()

    def save_model(self, round_num, path='out-fl'):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'model': self.model.state_dict(),
            'model_args': {
                'n_layer': self.model.config.n_layer,
                'n_head': self.model.config.n_head,
                'n_embd': self.model.config.n_embd,
                'block_size': self.model.config.block_size,
                'bias': self.model.config.bias,
                'vocab_size': self.model.config.vocab_size,
                'dropout': self.model.config.dropout
            },
            'round': round_num
        }
        
        # Save best model
        best_path = os.path.join(path, 'best_model.pt')
        torch.save(checkpoint, best_path)
        
        # Save round-specific checkpoint
        round_path = os.path.join(path, f'round_{round_num}_model.pt')
        torch.save(checkpoint, round_path)
        
        print(f"Model checkpoints saved to {path}")