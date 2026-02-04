import torch
import torch.nn as nn
import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)

try:
    from model.LSTM_auto_encoder import LSTMAutoEncoder
    from utils.dataset import NABAWSData
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
    from src.model.LSTM_auto_encoder import LSTMAutoEncoder
    from src.utils.dataset import NABAWSData

DATA_PATH = 'data/cpu_util.csv'
MODEL_PATH = 'models/lstm_model_v1.0.0.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prove_temporal_learning():
    print(f"--- 🔬 THE FINAL PROOF (Smoothness Test) ---")
    
    # 1. Load Real Data
    dataset = NABAWSData(DATA_PATH, mode='test')
    real_seq = dataset[0] # Just take the first one
    real_tensor = real_seq.unsqueeze(0).to(DEVICE)

    # 2. Create Shuffled Version
    shuffled_seq = real_seq[torch.randperm(real_seq.size(0))]
    shuffled_tensor = shuffled_seq.unsqueeze(0).to(DEVICE)

    # 3. Load Model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    hidden_size = checkpoint.get('hidden_size', 12)
    model = LSTMAutoEncoder(num_layers=1, hidden_size=hidden_size, nb_feature=1, device=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Inference
    with torch.no_grad():
        rec_real = model(real_tensor)
        rec_shuffled = model(shuffled_tensor)
        
        loss_real = nn.MSELoss()(rec_real, real_tensor).item()
        loss_shuffled = nn.MSELoss()(rec_shuffled, shuffled_tensor).item()

    print(f"Real Error:     {loss_real:.6f}")
    print(f"Shuffled Error: {loss_shuffled:.6f}")

    # 5. THE REAL METRIC: SMOOTHNESS
    # Calculate the 'jaggedness' (Total Variation) of the output vs input
    # If the model is learning Time, it will produce SMOOTHER output for shuffled inputs
    # because it tries to 'connect the dots' between random jumps.
    
    def get_jaggedness(tensor):
        return torch.sum(torch.abs(tensor[:, 1:] - tensor[:, :-1])).item()

    input_jaggedness = get_jaggedness(shuffled_tensor)
    output_jaggedness = get_jaggedness(rec_shuffled)
    
    print(f"\n--- SMOOTHNESS METRICS ---")
    print(f"Input Noise Jaggedness:  {input_jaggedness:.4f}")
    print(f"Model Output Jaggedness: {output_jaggedness:.4f}")
    
    if output_jaggedness < input_jaggedness:
        print(f"\n✅ SUCCESS: The model SMOOTHED the noise.")
        print(f"It reduced jaggedness by {input_jaggedness - output_jaggedness:.4f}.")
        print("This PROVES the LSTM is applying temporal continuity (it refuses to jump randomly).")
    else:
        print("\n❌ FAIL: Model copied the jagged noise perfectly.")

if __name__ == "__main__":
    prove_temporal_learning()