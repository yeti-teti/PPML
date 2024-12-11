import os
import torch
from model import GPTConfig, GPT
from privacy_utils import SecureAggregation

class PrivateServer:
    def __init__(self, noise_scale=0.01):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.init_model()
        self.secure_aggregator = SecureAggregation(noise_scale=noise_scale)
        print(f"Server initialized with secure aggregation")

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
        print("Models aggregated securely")

    def get_model_state(self):
        return self.model.state_dict()

    def save_model(self, round_num, path='out-fl'):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            'model': self.model.state_dict(),
            'model_args': self.model.config.__dict__,
            'round': round_num
        }
        checkpoint_path = os.path.join(path, 'ckpt.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved (Round {round_num})")