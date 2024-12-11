import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MedicalDataset(Dataset):
    def __init__(self, client_id, split='train', block_size=1024):
        self.data_dir = os.path.join('data/medical', f'client_{client_id}')
        data_file = os.path.join(self.data_dir, f'{split}.bin')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"No data file found at {data_file}")
            
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

def create_dataloader(client_id, split='train', batch_size=4, block_size=1024):
    dataset = MedicalDataset(client_id=client_id, split=split, block_size=block_size)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True
    )