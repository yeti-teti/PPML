import os
import time
import math
import pickle
import copy
from contextlib import nullcontext

import numpy as np
import torch
from model import GPTConfig, GPT

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, client_id, split='train', block_size=1024):
        """Initialize dataset for a specific client"""
        self.data_dir = os.path.join('data/medical_fl', f'client_{client_id}')
        data_file = os.path.join(self.data_dir, f'{split}.bin')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"No data file found at {data_file}. "
                "Please run prepare.py first to generate the data."
            )
            
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.length = len(self.data) - block_size

    def __len__(self):
        return max(self.length, 0)

    def __getitem__(self, idx):
        block_data = self.data[idx:idx + self.block_size]
        x = torch.from_numpy(block_data[:-1].astype(np.int64))
        y = torch.from_numpy(block_data[1:].astype(np.int64))
        return x, y

def create_dataloader(client_id, split, batch_size=4, block_size=1024):
    """Create a dataloader for a specific client"""
    dataset = MedicalDataset(client_id, split, block_size)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True
    )

def train_client(model, optimizer, dataloader, client_id, device, local_epochs=1):
    """Train model on a single client"""
    model.train()
    total_loss = 0
    iter_count = 0

    for epoch in range(local_epochs):
        for iter, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            logits, loss = model(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iter_count += 1

            if iter % 10 == 0:
                print(f"Client {client_id}, Epoch {epoch}, Iter {iter}: Loss = {loss.item():.4f}")

    return total_loss / iter_count if iter_count > 0 else float('inf')

def aggregate_models(models):
    """Aggregate models using FedAvg"""
    global_dict = models[0].state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.mean(torch.stack([
            model.state_dict()[key].float()
            for model in models
        ], dim=0), dim=0)
    return global_dict

def main():
    # Training Configuration
    num_clients = 3
    num_rounds = 5
    local_epochs = 1
    batch_size = 4
    block_size = 1024
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model Configuration
    model_args = dict(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=block_size,
        bias=False,
        vocab_size=50257,
        dropout=0.1
    )

    # Initialize global model
    global_model = GPT(GPTConfig(**model_args))
    global_model = global_model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in global_model.parameters()):,}")

    # Create dataloaders for each client
    train_loaders = {}
    val_loaders = {}
    for i in range(num_clients):
        train_loaders[i] = create_dataloader(i, 'train', batch_size, block_size)
        val_loaders[i] = create_dataloader(i, 'val', batch_size, block_size)

    # Training loop
    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")
        global_weights = global_model.state_dict()
        client_models = []

        # Train on each client
        for client_id in range(num_clients):
            print(f"\nTraining on Client {client_id}")
            
            # Initialize client model with global weights
            client_model = GPT(GPTConfig(**model_args))
            client_model.load_state_dict(global_weights)
            client_model = client_model.to(device)

            # Create optimizer
            optimizer = client_model.configure_optimizers(
                weight_decay=0.1,
                learning_rate=learning_rate,
                betas=(0.9, 0.95),
                device_type='cuda' if 'cuda' in device else 'cpu'
            )

            # Train client model
            loss = train_client(
                client_model,
                optimizer,
                train_loaders[client_id],
                client_id,
                device,
                local_epochs
            )
            print(f"Client {client_id} completed training. Average Loss: {loss:.4f}")
            client_models.append(client_model)

        # Aggregate models
        print("\nAggregating models...")
        global_model.load_state_dict(aggregate_models(client_models))

        # Save checkpoint
        print("Saving checkpoint...")
        checkpoint = {
            'model': global_model.state_dict(),
            'model_args': model_args,
            'round': round_num,
        }
        os.makedirs('out-fl', exist_ok=True)
        torch.save(checkpoint, os.path.join('out-fl', 'ckpt.pt'))

    print("\nFederated training completed!")

if __name__ == '__main__':
    main()