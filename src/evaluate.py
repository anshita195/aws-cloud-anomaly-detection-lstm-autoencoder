import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import sys
import os
import random

# Robust import
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from utils.dataset import NABAWSData
    from model.LSTM_auto_encoder import LSTMAutoEncoder
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
    from src.utils.dataset import NABAWSData
    from src.model.LSTM_auto_encoder import LSTMAutoEncoder

# CONFIG
DATA_PATH = 'data/cpu_util.csv'
MODEL_PATH = 'models/lstm_model_v1.0.0.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1 # Process one by one for precise injection

def inject_anomaly(sequence):
    """
    Simulates SUBTLE failures (Harder to detect).
    FIXED: Handles shape broadcasting correctly.
    """
    seq = sequence.clone()
    anomaly_type = random.choice(['drift', 'freeze', 'subtle_noise'])
    
    # 50% chance to be an anomaly
    if random.random() > 0.5:
        return seq, 0 # Normal
    
    # --- Stealth Chaos Injection ---
    if anomaly_type == 'drift':
        # Harder: Gradual memory leak. 
        # Create a linear increase from 0 to 0.6
        drift = torch.linspace(0, 0.6, seq.shape[0]).to(seq.device)
        
        # CRITICAL FIX: Reshape drift from [50] to [50, 1] to match sequence
        drift = drift.view(-1, 1) 
        
        seq += drift
        
    elif anomaly_type == 'freeze':
        # Harder: Freezes at a "Normal" value (e.g., 0.4)
        seq[:] = 0.4 
        
    elif anomaly_type == 'subtle_noise':
        # Harder: Adds just a little noise (0.3 SD instead of 2.0)
        seq += torch.randn_like(seq) * 0.3
        
    return seq, 1 # Is Anomaly

def evaluate():
    print(f"--- 🧪 Starting CHAOS TESTING on {DEVICE} ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data not found at {DATA_PATH}")
        return

    # 1. Load Data
    test_dataset = NABAWSData(DATA_PATH, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    sample = test_dataset[0]
    model = LSTMAutoEncoder(num_layers=1, hidden_size=64, nb_feature=sample.shape[1], device=DEVICE)
    
    # Smart Load
    paths = [MODEL_PATH, 'models/lstm_model.pth']
    loaded = False
    for p in paths:
        if os.path.exists(p):
            checkpoint = torch.load(p, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            print(f"✅ Model loaded from {p}")
            loaded = True
            break
    if not loaded: return

    # 3. Chaos Inference Loop
    losses = []
    ground_truth = []
    
    print("--- 🌪️ Injecting Synthetic Failures (Spikes, Flatlines, Noise) ---")
    with torch.no_grad():
        for batch in test_loader:
            # Inject Chaos
            batch_data = batch[0] # Unpack if batch size 1
            corrupted_data, is_anomaly = inject_anomaly(batch_data)
            
            # Prepare Input
            input_tensor = corrupted_data.view(1, 50, 1).to(DEVICE)
            ground_truth.append(is_anomaly)
            
            # Forward Pass
            output = model(input_tensor)
            loss = nn.MSELoss()(output, input_tensor).item()
            losses.append(loss)

    losses = np.array(losses)
    ground_truth = np.array(ground_truth)
    
    # 4. Realistic Metrics
    # Calculate threshold on just the NORMAL data to simulate real production calibration
    normal_losses = losses[ground_truth == 0]
    threshold = np.mean(normal_losses) + 3 * np.std(normal_losses) # 3-Sigma Rule
    
    predictions = (losses > threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='binary')
    
    print(f"\n--- 📊 REALISTIC Results ---")
    print(f"Dynamic Threshold (3-Sigma of Normal): {threshold:.6f}")
    print(f"Precision: {precision:.4f} (How many alerts were real?)")
    print(f"Recall:    {recall:.4f} (Did we catch all the crashes?)")
    print(f"F1-Score:  {f1:.4f}")

    # 5. Plotting
    plt.figure(figsize=(12, 6))
    
    # Histogram - Showing Separation
    plt.subplot(1, 2, 1)
    sns.histplot(losses[ground_truth==0], bins=50, color='green', alpha=0.5, label='Normal')
    sns.histplot(losses[ground_truth==1], bins=50, color='red', alpha=0.5, label='Injected Anomalies')
    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
    plt.title('Separation of Normal vs Anomaly')
    plt.legend()
    plt.yscale('log') # Log scale to see the massive spikes
    
    # ROC
    fpr, tpr, _ = roc_curve(ground_truth, losses)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Realistic ROC Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Evaluation_Report.png')
    print("✅ Saved 'Evaluation_Report.png'")

if __name__ == "__main__":
    evaluate()