import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sys
import os
import logging
logging.basicConfig(filename='inference.log', level=logging.INFO)

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from model.LSTM_auto_encoder import LSTMAutoEncoder
except ImportError:
    # Fallback if running from root
    sys.path.append("/content/AWS_Cloud_Anomaly/src")
    from model.LSTM_auto_encoder import LSTMAutoEncoder

app = FastAPI(title="AWS CPU Anomaly Detector", version="1.0")

# --- CONFIG ---
MODEL_PATH = "../models/lstm_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.0128 

# --- LOAD MODEL ---
model = LSTMAutoEncoder(num_layers=1, hidden_size=64, nb_feature=1, device=DEVICE)

print("Loading model...")
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model load error: {e}")
else:
    print(f"⚠️ WARNING: Model file not found at {MODEL_PATH}")

class CPUSequence(BaseModel):
    sequence: List[float]

@app.post("/predict")
def predict(data: CPUSequence):
    input_list = data.sequence
    if len(input_list) != 50:
        raise HTTPException(status_code=400, detail=f"Expected 50 data points, got {len(input_list)}")
    
    input_tensor = torch.FloatTensor(input_list).view(1, 50, 1).to(DEVICE)
    
    with torch.no_grad():
        reconstruction = model(input_tensor)
        loss = torch.nn.MSELoss()(reconstruction, input_tensor).item()
    
    is_anomaly = loss > THRESHOLD
    logging.info(f"Input processed. Loss: {loss} | Anomaly: {is_anomaly}")
    return {
        "reconstruction_error": round(loss, 6),
        "threshold": THRESHOLD,
        "is_anomaly": is_anomaly,
        "status": "CRITICAL" if is_anomaly else "NORMAL"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
