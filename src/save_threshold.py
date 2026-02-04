import torch
import os
import sys

# Setup imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.LSTM_auto_encoder import LSTMAutoEncoder

# CONFIG
MODEL_PATH = 'models/lstm_model_v1.0.0.pth'
# PASTE YOUR THRESHOLD FROM STEP 2 HERE 👇
THRESHOLD_VALUE = 0.015043  

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH)
    checkpoint['threshold'] = THRESHOLD_VALUE
    # Ensure we save the hidden_size too so we don't forget it
    checkpoint['hidden_size'] = 12 
    torch.save(checkpoint, MODEL_PATH)
    print(f"✅ Saved Threshold {THRESHOLD_VALUE} and Size 12 to model artifact.")
else:
    print("❌ Model not found.")