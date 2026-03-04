# AWS Chaos-Validated Cloud Anomaly Detection

## Overview

This is a production-grade MLOps system designed for unsupervised monitoring of cloud infrastructure. It utilizes an LSTM Autoencoder architecture to learn baseline temporal patterns in AWS EC2 CPU utilization and identifies anomalies through reconstruction error analysis.

## Core Engineering Features

* **Unsupervised Learning**: Employs an encoder-decoder LSTM to learn normal server behavior without requiring labeled historical anomaly data.
* **Chaos Engineering Validation**: Includes an integrated chaos generator to stress-test the model against simulated infrastructure failures such as memory leaks, process hangs, and sensor degradation.
* **MLOps Architecture**: Utilizes a professional directory structure that separates production API code, core model logic, and serialized artifacts.
* **Real-time Observability**: Fully instrumented with Prometheus and Grafana for production-level monitoring and alerting.

## System Architecture

The model utilizes a PyTorch-based LSTM Autoencoder. A sequence of 50 CPU utilization data points is compressed into a 12-unit latent space by the encoder and subsequently reconstructed by the decoder.

* **Reconstruction Logic**: Anomalies are detected when the Mean Squared Error (MSE) between the input and reconstructed sequence exceeds a dynamic statistical threshold.
* **Temporal Integrity Proof**: The system includes validation logic to prove that the LSTM enforces temporal continuity rather than simple data memorization.

## Reliability and Validation

To ensure production readiness, the model is subjected to rigorous reliability testing rather than relying on standard accuracy metrics.

### Chaos Injection Suite

The `src/evaluate.py` script simulates specific infrastructure failure modes to verify detection robustness:

* **Drift**: Simulates gradual resource exhaustion or memory leaks.
* **Freeze**: Simulates hung processes by forcing static metric values.
* **Noise**: Simulates hardware degradation through high-variance jitter.

### Performance Results

The system is tuned using a 1.8-Sigma dynamic threshold to balance sensitivity with operational reliability. This configuration prioritizes high precision to minimize alert fatigue for DevOps teams. Detailed performance artifacts are stored in `metrics.json` upon evaluation.

## MLOps and Observability

The project is containerized and orchestrated to provide a complete monitoring solution.

### Components

* **Inference API**: A FastAPI-based service providing a high-performance `/predict` endpoint for real-time anomaly detection.
* **Prometheus**: Collects real-time metrics including `deepguard_anomalies_total` and `deepguard_reconstruction_error`.
* **Grafana**: Provides a visual dashboard for monitoring infrastructure health and anomaly occurrences.
* **Docker Compose**: Orchestrates the API, database, and visualization layers for one-click deployment.

## Getting Started

### Installation

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt

```


2. Train the model:
```bash
python src/main.py

```


3. Execute Chaos Engineering evaluation:
```bash
python src/evaluate.py

```



### Docker Deployment

Launch the complete monitoring stack:

```bash
docker-compose up --build

```

The API documentation can be accessed at `http://localhost:8000/docs`.