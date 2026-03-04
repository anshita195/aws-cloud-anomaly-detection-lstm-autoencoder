# AWS Chaos-Validated Cloud Anomaly Detection

## Live Demo and Observability

The system is deployed and available for real-time testing:

* **Interactive API Documentation (Swagger UI)**: [https://aws-cloud-anomaly-detection-lstm.onrender.com/docs](https://www.google.com/search?q=https://aws-cloud-anomaly-detection-lstm.onrender.com/docs)
* **Live Prometheus Metrics**: [https://aws-cloud-anomaly-detection-lstm.onrender.com/metrics](https://www.google.com/search?q=https://aws-cloud-anomaly-detection-lstm.onrender.com/metrics)

## Live Testing Instructions

Manual verification of the detection capabilities can be performed using the interactive Swagger UI.

### Step-by-Step Guide

1. Navigate to the /docs URL provided above.
2. Expand the **POST /predict** endpoint.
3. Click **Try it out**.
4. Paste one of the example arrays below into the `sequence` field of the Request Body.
5. Click **Execute** and review the JSON response.

### Example 1: Normal Baseline (Stable System)

This array simulates a server at idle or steady load (average approximately 12% CPU).

* **Expected Result**: `is_anomaly: false`
* **Explanation**: The model successfully reconstructs this known pattern, resulting in a reconstruction error below the threshold.

```json
{
  "sequence": [
    0.11, 0.12, 0.11, 0.13, 0.12, 0.11, 0.12, 0.14, 0.12, 0.11,
    0.13, 0.12, 0.11, 0.12, 0.11, 0.13, 0.12, 0.11, 0.14, 0.12,
    0.11, 0.13, 0.12, 0.11, 0.12, 0.11, 0.13, 0.12, 0.11, 0.12,
    0.11, 0.14, 0.12, 0.11, 0.13, 0.12, 0.11, 0.12, 0.11, 0.13,
    0.12, 0.11, 0.12, 0.14, 0.12, 0.11, 0.13, 0.12, 0.11, 0.12
  ]
}

```

### Example 2: Anomalous Drift (Resource Leak)

This array simulates a steady climb in utilization (10% to 99%), typical of a memory leak or runaway process.

* **Expected Result**: `is_anomaly: true`
* **Explanation**: This trend deviates significantly from baseline patterns, causing the reconstruction error to spike and cross the detection threshold.

```json
{
  "sequence": [
    0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28,
    0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48,
    0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68,
    0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88,
    0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99
  ]
}

```

## Understanding the Output

The API returns a JSON response with three key fields:

1. **`reconstruction_error`**: The Mean Squared Error (MSE) between the input data and the model's reconstruction. Higher values indicate higher abnormality.
2. **`threshold`**: The current dynamic limit (1.8-Sigma: 0.013664). Any error above this triggers an alert.
3. **`is_anomaly`**: A boolean flag where `true` confirms a detected failure.

## Core Engineering Features

* **Unsupervised Temporal Learning**: Employs an encoder-decoder LSTM to learn normal server behavior without requiring labeled historical anomaly data.
* **Integrated Training Pipeline**: The training process is optimized with early stopping to prevent overfitting, automated checkpointing to save only the best model version, and loss history tracking for convergence verification.
* **Chaos Engineering Validation**: Includes a dedicated chaos generator to stress-test detection capabilities against simulated infrastructure failures such as memory leaks (drift), process hangs (freeze), and sensor degradation (noise).
* **Production-Ready Observability**: Fully instrumented with Prometheus and Grafana for real-time production monitoring and automated alerting.

## System Architecture

The model utilizes a PyTorch-based LSTM Autoencoder where a sequence of 50 CPU utilization data points is compressed into a 12-unit latent space and subsequently reconstructed.

* **Detection Logic**: Anomalies are flagged when the Mean Squared Error (MSE) between the input and reconstructed sequence exceeds a dynamic statistical threshold.
* **Temporal Integrity Verification**: The system includes specific validation logic to confirm the model captures temporal continuity rather than simple data memorization.

## Reliability and Validation

To ensure operational readiness, the model is subjected to rigorous reliability testing rather than relying on standard accuracy metrics.

### Chaos Injection Suite

The evaluation suite simulates specific infrastructure failure modes to verify detection robustness:

* **Drift**: Simulates gradual resource exhaustion or creeping memory leaks.
* **Freeze**: Simulates hung processes or sensor failures by forcing static metric values.
* **Noise**: Simulates hardware degradation or communication jitter.

### Performance Benchmarks

The system is tuned using a 1.8-Sigma dynamic threshold to balance sensitivity with operational reliability. This configuration prioritizes high precision to minimize alert fatigue for DevOps teams.

* **Precision**: 0.8493
* **Recall**: 0.6425
* **F1-Score**: 0.7316

## MLOps and Observability

The project is fully containerized and orchestrated to provide a complete infrastructure monitoring solution.

### Components

* **Inference API**: A FastAPI service providing a high-performance `/predict` endpoint for real-time anomaly detection.
* **Prometheus**: Collects real-time metrics including `deepguard_anomalies_total` and `deepguard_reconstruction_error`.
* **Grafana**: Provides visual dashboards for monitoring infrastructure health and anomaly occurrences.
* **Docker Compose**: Orchestrates the API, monitoring, and visualization layers for standardized deployment.

## Getting Started

### Installation and Local Execution

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt

```

2. Train the model with integrated utilities (Early Stopping, Best-Model Checkpointing):

```bash
python src/main.py

```

3. Execute Chaos Engineering evaluation:

```bash
python src/evaluate.py

```

### Production Deployment

Launch the complete monitoring stack via Docker Compose:

```bash
docker-compose up --build

```

The API documentation and `/predict` endpoint will be available at `http://localhost:8000/docs`.