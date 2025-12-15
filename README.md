# 🫀 Heart Disease Prediction API (MLP from Scratch)

This project is an **end-to-end Machine Learning pipeline** that implements a
**Multi-Layer Perceptron (MLP) neural network from scratch** using NumPy and
deploys it as a **REST API using FastAPI**.

The goal is to demonstrate both **core ML understanding** and **production-ready deployment skills**.

---

## 🚀 Project Highlights

- MLP implemented **from scratch** (no TensorFlow / PyTorch)
- Manual forward & backward propagation
- Feature engineering and normalization
- Model serialization and loading
- REST API for real-time inference
- Clean and modular project structure

---

## 🧠 Model Details

- Architecture: Fully-connected MLP
- Activation function: Sigmoid
- Loss: Binary classification
- Output: Probability of heart disease
- Test Accuracy: ~87%

---

## ⚙️ Installation & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run

```bash
uvicorn app.main:app --reload
```

You can open the interactive API docs at : http://127.0.0.1:8000/docs

## Prediction Example

### Request

```json
{
"features": [40, 1, 140, 289, 0, 172, 0, 0.0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
}
```

### Response

```json
{
"prediction": 0,
"probability": 0.1
}
```
