# AWS Chaos-Validated Cloud Anomaly Detection

## Overview

This is a production-grade MLOps system developed for unsupervised monitoring of cloud infrastructure. It utilizes an LSTM Autoencoder architecture to learn baseline temporal patterns in AWS EC2 CPU utilization and identifies anomalies through reconstruction error analysis.

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