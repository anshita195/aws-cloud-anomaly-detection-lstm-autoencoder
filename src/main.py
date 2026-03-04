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
    # Import the utility classes
    from utils.callbacks import EarlyStopping
    from utils.model_management import ModelManagement
    from utils.plot_class import LossCheckpoint
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
    from src.model.LSTM_auto_encoder import LSTMAutoEncoder
    from src.utils.dataset import NABAWSData
    from src.utils.callbacks import EarlyStopping
    from src.utils.model_management import ModelManagement
    from src.utils.plot_class import LossCheckpoint

# 2. CONFIG
DATA_PATH = 'data/cpu_util.csv'
MODELS_DIR = 'models/'
MODEL_NAME = 'lstm_model_v1.0.0'
HIDDEN_SIZE = 12
BATCH_SIZE = 64
EPOCHS = 100       # Increased because EarlyStopping will handle the exit
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    print(f"--- Starting Integrated Training (Hidden Size={HIDDEN_SIZE}) ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    train_dataset = NABAWSData(DATA_PATH, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMAutoEncoder(num_layers=1, hidden_size=HIDDEN_SIZE, nb_feature=1, device=DEVICE)
    model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # 3. INITIALIZE UTILITIES
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    
    # Model manager will save the best model to models/lstm_model_v1.0.0.pth
    model_manager = ModelManagement(path=MODELS_DIR, name_model=MODEL_NAME)
    
    # Early stopping will break the loop if loss doesn't improve for 5 epochs
    early_stopper = EarlyStopping(patience=5)
    
    # Loss plotter will record loss history
    loss_plotter = LossCheckpoint()
    
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
        
        # RECORD LOSS
        loss_plotter.losses.append(avg_loss)
        
        # CHECKPOINT (Saves only if avg_loss is the best seen so far)
        model_manager.checkpoint(epoch, model, optimizer, avg_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

        # CHECK EARLY STOPPING
        if early_stopper.check_training(avg_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training Complete. Best model saved in {MODELS_DIR}")
    
    # FINAL VISUALIZATION
    loss_plotter.plot(log=True)

if __name__ == "__main__":
    train()