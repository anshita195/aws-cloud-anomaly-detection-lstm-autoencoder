import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# --- FIXED IMPORTS (Removed 'src.') ---
from utils.dataset import NABAWSData 
from model.LSTM_auto_encoder import LSTMAutoEncoder
from utils.callbacks import EarlyStopping
from utils.model_management import ModelManagement
from utils.plot_class import LossCheckpoint

# Config
name_model = 'lstm_model'
path_model = '../models/'
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
csv_path = '../data/cpu_util.csv'

print(f"Using Device: {device}")

# Check if CSV exists before starting
import os
if not os.path.exists(csv_path):
    print(f"❌ ERROR: File not found at {csv_path}")
    print("⚠️ PLEASE UPLOAD 'cpu_util.csv' TO THE 'AWS_Cloud_Anomaly/data' FOLDER!")
    exit()

print("Loading Data...")
train_seq = NABAWSData(csv_path, mode='train')
valid_seq = NABAWSData(csv_path, mode='val')
test_seq = NABAWSData(csv_path, mode='test')

train_loader = DataLoader(train_seq, batch_size, shuffle=True)
valid_loader = DataLoader(valid_seq, batch_size, shuffle=True)
test_loader = DataLoader(test_seq, batch_size, shuffle=False)

sample = train_seq[0]
seq_length = sample.shape[0]
nb_feature = sample.shape[1]
print(f"Sequence Length: {seq_length}, Features: {nb_feature}")

model = LSTMAutoEncoder(num_layers=1, hidden_size=64, nb_feature=nb_feature, device=device)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

earlyStopping = EarlyStopping(patience=5)
model_management = ModelManagement(path_model, name_model)
loss_checkpoint_train = LossCheckpoint()
loss_checkpoint_valid = LossCheckpoint()

def train(epoch):
    model.train()
    train_loss = 0
    for id_batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model.forward(data)
        loss = criterion(data, output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, avg_loss))
    loss_checkpoint_train.losses.append(avg_loss)

def evaluate(loader, validation=False, epoch=0):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for id_batch, data in enumerate(loader):
            data = data.to(device)
            output = model.forward(data)
            loss = criterion(data, output)
            eval_loss += loss.item()
            
    avg_loss = eval_loss / len(loader)
    print('====> Validation Average loss: {:.6f}'.format(avg_loss))
    
    if validation:
        loss_checkpoint_valid.losses.append(avg_loss)
        model_management.checkpoint(epoch, model, optimizer, avg_loss)
        return earlyStopping.check_training(avg_loss)

if __name__ == "__main__":
    for epoch in range(1, 21):
        train(epoch)
        if evaluate(valid_loader, validation=True, epoch=epoch):
            print("Early Stopping Triggered")
            break
    loss_checkpoint_train.plot()
