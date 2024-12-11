import torch
from model import GPTConfig, GPT
from dataset import create_dataloader
from privacy_utils import PrivacyEngine, mixup_data

class PrivateClient:
    def __init__(self, 
                 client_id, 
                 batch_size=4, 
                 block_size=1024,
                 noise_multiplier=1.0,
                 max_grad_norm=1.0):
        self.client_id = client_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.block_size = block_size
        
        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine(
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm
        )
        
        # Initialize model and data
        self.model = self.init_model()
        self.setup_data()
        print(f"Client {client_id} initialized with privacy protection")

    def init_model(self):
        model_args = dict(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=self.block_size,
            bias=False,
            vocab_size=50257,
            dropout=0.1
        )
        model = GPT(GPTConfig(**model_args))
        return model.to(self.device)

    def setup_data(self):
        self.train_loader = create_dataloader(
            self.client_id, 'train', 
            self.batch_size, self.block_size
        )
        self.val_loader = create_dataloader(
            self.client_id, 'val', 
            self.batch_size, self.block_size
        )

    def train_epoch(self):
        self.model.train()
        optimizer = self.model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-4,
            betas=(0.9, 0.95),
            device_type='cuda' if 'cuda' in self.device else 'cpu'
        )
    
        total_loss = 0
        num_batches = 0
    
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to device first
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply mixup augmentation only for continuous data, not for discrete tokens
            # Since we're working with token indices, we'll skip mixup
            # data, target = mixup_data(data, target)  # Remove this line
    
            optimizer.zero_grad()
            output, loss = self.model(data, target)
            loss.backward()
    
            # Apply differential privacy
            self.privacy_engine.clip_gradients(self.model)
            self.privacy_engine.add_noise_to_gradients(self.model)
    
            optimizer.step()
    
            total_loss += loss.item()
            num_batches += 1
    
            if batch_idx % 10 == 0:
                print(f'Client {self.client_id}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, loss = self.model(data, target)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss

    def get_model_state(self):
        return self.model.state_dict()

    def update_model(self, model_state):
        self.model.load_state_dict(model_state)