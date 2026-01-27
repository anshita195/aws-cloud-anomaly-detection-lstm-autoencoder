import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class NABAWSData(Dataset):
    def __init__(self, path, mode='train', window_size=50):
        self.window_size = window_size
        df = pd.read_csv(path)
        data = df['value'].values.reshape(-1, 1)
        self.scaler = MinMaxScaler()
        train_len = int(len(data) * 0.6)
        self.scaler.fit(data[:train_len])
        data_normalized = self.scaler.transform(data)
        
        n = len(data_normalized)
        if mode == 'train':
            self.data = data_normalized[:int(n*0.6)]
        elif mode == 'val':
            self.data = data_normalized[int(n*0.6):int(n*0.8)]
        elif mode == 'test':
            self.data = data_normalized[int(n*0.8):]
            
        self.sequences = self._create_sequences(self.data)
        self.sequences = torch.FloatTensor(self.sequences)

    def _create_sequences(self, data):
        seqs = []
        for i in range(len(data) - self.window_size):
            seq = data[i:i + self.window_size]
            seqs.append(seq)
        return np.array(seqs)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
