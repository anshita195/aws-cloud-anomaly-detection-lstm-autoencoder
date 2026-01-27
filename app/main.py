import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from typing import List
import sys
import os
import logging
from pythonjsonlogger import jsonlogger
import time

# --- 1. SETUP STRUCTURED LOGGING (Industry Standard) ---
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# --- 2. SETUP PATHS & IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
try:
    from model.LSTM_auto_encoder import LSTMAutoEncoder
except ImportError:
    # Fallback
    sys.path.append("/app/src")
    from model.LSTM_auto_encoder import LSTMAutoEncoder

app = FastAPI(title="AWS CPU Anomaly Detector", version="1.0.0")

# --- 3. PROMETHEUS METRICS (The "Eyes" of the System) ---
# Custom metric: Count how many anomalies we detect
ANOMALY_COUNTER = Counter('deepguard_anomalies_total', 'Total number of anomalies detected')
# Custom metric: Track reconstruction error distribution
RECONSTRUCTION_ERROR = Histogram('deepguard_reconstruction_error', 'MSE Loss of the model')

# Instrument the app (Standard latency/request count metrics)
Instrumentator().instrument(app).expose(app)

# --- CONFIG ---
MODEL_PATH = "../models/lstm_model_v1.0.0.pth" # Using the versioned name
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.0128 

# --- LOAD MODEL ---
model = LSTMAutoEncoder(num_layers=1, hidden_size=64, nb_feature=1, device=DEVICE)
logger.info(f"Initializing model on {DEVICE}")

if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Model load error: {str(e)}")
else:
    logger.warning(f"⚠️ Model file not found at {MODEL_PATH}")

class CPUSequence(BaseModel):
    sequence: List[float]

@app.post("/predict")
def predict(data: CPUSequence):
    start_time = time.time()
    input_list = data.sequence
    
    if len(input_list) != 50:
        logger.error("Invalid input length", extra={"length": len(input_list)})
        raise HTTPException(status_code=400, detail=f"Expected 50 data points, got {len(input_list)}")
    
    input_tensor = torch.FloatTensor(input_list).view(1, 50, 1).to(DEVICE)
    
    with torch.no_grad():
        reconstruction = model(input_tensor)
        loss = torch.nn.MSELoss()(reconstruction, input_tensor).item()
    
    is_anomaly = loss > THRESHOLD
    
    # Update Metrics
    RECONSTRUCTION_ERROR.observe(loss)
    if is_anomaly:
        ANOMALY_COUNTER.inc()
        logger.warning("Anomaly Detected", extra={"loss": loss, "threshold": THRESHOLD})
    else:
        logger.info("Normal Traffic", extra={"loss": loss})
        
    return {
        "reconstruction_error": round(loss, 6),
        "threshold": THRESHOLD,
        "is_anomaly": is_anomaly,
        "processing_time": round(time.time() - start_time, 4)
    }

@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}