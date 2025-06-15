# ⚡ Distributed Transformers Training with GPU on Vertex AI

This repository contains a full pipeline for **distributed fine-tuning of transformer models** using **PyTorch**, **Hugging Face Accelerate**, and **Vertex AI**.  
The training is GPU-enabled and orchestrated through **Kubeflow Pipelines (KFP)** for scalable execution on Google Cloud.

---

## 🚀 Features

- Distributed model training using Hugging Face’s `accelerate`  
- Multi-step pipeline with training and evaluation components  
- Automatic use of **NVIDIA GPUs** if available (e.g., T4)  
- Training and evaluation executed in isolated containers  
- Model and results saved and loaded from **Google Cloud Storage (GCS)**  
- Pipeline execution via **Vertex AI Pipelines** or as a standalone `CustomJob`  
- Integrated classification metrics (`precision`, `recall`, `f1`) with `scikit-learn`

---

## ☁️ Vertex AI Deployment

### 🔧 Build and Push Docker Image

```bash
docker build -t train-distributed .
docker tag train-distributed us-central1-docker.pkg.dev/YOUR_PROJECT/my-kfp-repo/train-distributed:latest
docker push us-central1-docker.pkg.dev/YOUR_PROJECT/my-kfp-repo/train-distributed:latest
```

### 1. Compile the Pipeline

```bash
python compile_pipeline.py
```

### 2. Launch the Pipeline

```bash
python run_pipeline.py
```

---
