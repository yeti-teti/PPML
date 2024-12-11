import os
import torch
from model import GPTConfig, GPT

class FederatedServer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.init_model()
        print(f"Server initialized (using device: {self.device})")
        
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
        """Aggregate models using FedAvg"""
        global_dict = self.model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.mean(torch.stack([
                client_model[key].float() for client_model in client_models
            ], dim=0), dim=0)
        self.model.load_state_dict(global_dict)
        print("Models aggregated successfully")
    
    def get_model_state(self):
        return self.model.state_dict()
    
    def save_model(self, round_num, path='out-fl'):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            'model': self.model.state_dict(),
            'model_args': self.model.config.__dict__,
            'round': round_num
        }
        checkpoint_path = os.path.join(path, 'ckpt.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved (Round {round_num})")