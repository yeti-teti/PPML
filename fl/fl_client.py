import torch
from model import GPTConfig, GPT
from dataset import create_dataloader

class FederatedClient:
    def __init__(self, client_id, batch_size=4, block_size=1024):
        self.client_id = client_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.block_size = block_size
        self.model = self.init_model()
        self.setup_data()
        print(f"Client {client_id} initialized (using device: {self.device})")
        
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
        try:
            self.train_loader = create_dataloader(
                client_id=self.client_id,
                split='train',
                batch_size=self.batch_size,
                block_size=self.block_size
            )
            self.val_loader = create_dataloader(
                client_id=self.client_id,
                split='val',
                batch_size=self.batch_size,
                block_size=self.block_size
            )
            print(f"Data loaded for client {self.client_id}")
        except Exception as e:
            print(f"Error loading data for client {self.client_id}: {e}")
            raise
    
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
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output, loss = self.model(data, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Client {self.client_id}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Client {self.client_id} Average Training Loss: {avg_loss:.6f}")
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output, loss = self.model(data, target)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Client {self.client_id} Validation Loss: {avg_loss:.6f}")
        return avg_loss
    
    def get_model_state(self):
        return self.model.state_dict()
    
    def update_model(self, model_state):
        self.model.load_state_dict(model_state)