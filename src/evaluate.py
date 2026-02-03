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

# --- 1. ROBUST IMPORT FIX ---
# Add the directory containing this script (src/) to the Python path
# This allows us to import 'utils' and 'model' directly
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from utils.dataset import NABAWSData
    from model.LSTM_auto_encoder import LSTMAutoEncoder
except ImportError:
    # Fallback: If running as a module or from root with different structure
    from src.utils.dataset import NABAWSData
    from src.model.LSTM_auto_encoder import LSTMAutoEncoder

# --- CONFIG ---
# Paths are relative to where the command is run. 
# We assume running from Project Root: python src/evaluate.py
DATA_PATH = 'data/cpu_util.csv'
MODEL_PATH = 'models/lstm_model.pth' # Ensure this matches your saved model name
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

def evaluate():
    print(f"--- 🔍 Starting Evaluation on {DEVICE} ---")
    
    # Check paths
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data not found at {DATA_PATH}. Run from project root!")
        return
        
    # 1. Load Data (Test Set)
    test_dataset = NABAWSData(DATA_PATH, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    sample = test_dataset[0]
    nb_feature = sample.shape[1]
    
    model = LSTMAutoEncoder(num_layers=1, hidden_size=64, nb_feature=nb_feature, device=DEVICE)
    
    # Dynamic Model Loading
    # Try the specific version first, then fallback to generic
    paths_to_try = [MODEL_PATH, 'models/lstm_model_v1.0.0.pth', '../models/lstm_model.pth']
    loaded = False
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(DEVICE)
                model.eval()
                print(f"✅ Model loaded from {path}")
                loaded = True
                break
            except:
                continue
    
    if not loaded:
        print(f"❌ Model not found. Checked: {paths_to_try}")
        return

    # 3. Inference
    losses = []
    print("--- 🧠 Running Inference ---")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            output = model(batch)
            loss = nn.MSELoss(reduction='none')(output, batch)
            batch_loss = torch.mean(loss, dim=[1, 2]).cpu().numpy()
            losses.extend(batch_loss)

    losses = np.array(losses)
    
    # 4. Threshold & Metrics
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    threshold = mean_loss + 2 * std_loss
    
    print(f"\n--- 📊 Results ---")
    print(f"Threshold (Mean + 2σ): {threshold:.6f}")
    
    # Proxy Labels for Demonstration
    threshold_proxy = np.percentile(losses, 95)
    ground_truth = (losses > threshold_proxy).astype(int)
    predictions = (losses > threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='binary')
    
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    
    # 5. Plotting
    plt.figure(figsize=(12, 6))
    
    # Hist
    plt.subplot(1, 2, 1)
    sns.histplot(losses, bins=50, kde=True, color='blue')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Reconstruction Error Distribution')
    
    # ROC
    fpr, tpr, _ = roc_curve(ground_truth, losses)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('ROC Curve (Proxy Labels)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Evaluation_Report.png')
    print("✅ Saved 'Evaluation_Report.png'")

if __name__ == "__main__":
    evaluate()