import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# 1. SETUP PATHS
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)

try:
    from model.LSTM_auto_encoder import LSTMAutoEncoder
    from utils.dataset import NABAWSData
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
    from src.model.LSTM_auto_encoder import LSTMAutoEncoder
    from src.utils.dataset import NABAWSData

# 2. CONFIG (The Goldilocks Settings)
DATA_PATH = 'data/cpu_util.csv'
MODEL_PATH = 'models/lstm_model_v1.0.0.pth' 
HIDDEN_SIZE = 12  # Small enough to compress, big enough to learn
BATCH_SIZE = 64
EPOCHS = 30       # Give it time to learn patterns
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    print(f"--- 🚀 Starting Regularized Training (Hidden Size={HIDDEN_SIZE}) ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    train_dataset = NABAWSData(DATA_PATH, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMAutoEncoder(num_layers=1, hidden_size=HIDDEN_SIZE, nb_feature=1, device=DEVICE)
    model.to(DEVICE)
    
    # CRITICAL FIX: weight_decay=1e-5 prevents "Copy-Paste" memorization
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            output = model(batch)
            loss = criterion(output, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

    if not os.path.exists('models'): os.makedirs('models')
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': HIDDEN_SIZE,
        'threshold': 0.015 
    }, MODEL_PATH)
    
    print(f"✅ Training Complete. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()