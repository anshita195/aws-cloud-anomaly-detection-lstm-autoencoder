import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from typing import List
import sys
import os
import logging
from pythonjsonlogger import jsonlogger
import time

# --- LOGGING ---
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# --- SMART PATH SETUP ---
# This ensures imports work whether running from root or app/ folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

try:
    from src.model.LSTM_auto_encoder import LSTMAutoEncoder
except ImportError:
    from model.LSTM_auto_encoder import LSTMAutoEncoder

app = FastAPI(title="AWS CPU Anomaly Detector", version="1.1.0")

# --- METRICS ---
ANOMALY_COUNTER = Counter('deepguard_anomalies_total', 'Total anomalies')
RECONSTRUCTION_ERROR = Histogram('deepguard_reconstruction_error', 'MSE Loss')
Instrumentator().instrument(app).expose(app)

# --- ROBUST MODEL LOADING ---
# This finds the model folder relative to this script file
MODEL_PATH = os.path.join(parent_dir, 'models', 'lstm_model_v1.0.0.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GLOBAL VARIABLES
model = None
DYNAMIC_THRESHOLD = 0.015  # Default fallback

def load_model_and_stats():
    global model, DYNAMIC_THRESHOLD
    # IMPORTANT: hidden_size must match your training (12)
    model = LSTMAutoEncoder(num_layers=1, hidden_size=12, nb_feature=1, device=DEVICE)
    
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
            
            if 'threshold' in checkpoint:
                DYNAMIC_THRESHOLD = float(checkpoint['threshold'])
                logger.info(f"✅ Loaded Dynamic Threshold: {DYNAMIC_THRESHOLD:.6f}")
        except Exception as e:
            logger.error(f"❌ Model load error: {e}")
            # If size mismatch occurs, it prints the error here
    else:
        logger.error(f"❌ Model missing at {MODEL_PATH}")

load_model_and_stats()

class CPUSequence(BaseModel):
    sequence: List[float]

@app.post("/predict")
def predict(data: CPUSequence):
    start_time = time.time()
    input_list = data.sequence
    
    if len(input_list) != 50:
        raise HTTPException(status_code=400, detail="Expected 50 data points")
    
    input_tensor = torch.FloatTensor(input_list).view(1, 50, 1).to(DEVICE)
    
    with torch.no_grad():
        reconstruction = model(input_tensor)
        loss = torch.nn.MSELoss()(reconstruction, input_tensor).item()
    
    is_anomaly = loss > DYNAMIC_THRESHOLD
    
    RECONSTRUCTION_ERROR.observe(loss)
    if is_anomaly:
        ANOMALY_COUNTER.inc()
        logger.warning("Anomaly Detected", extra={"loss": loss, "threshold": DYNAMIC_THRESHOLD})
    
    return {
        "reconstruction_error": round(loss, 6),
        "threshold": round(DYNAMIC_THRESHOLD, 6),
        "is_anomaly": is_anomaly,
        "processing_time": round(time.time() - start_time, 4)
    }

@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.1.0"}