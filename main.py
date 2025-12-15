from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from utils import forward_pass

app = FastAPI(title="Heart Disease MLP API", version="1.0")

# ---- Load model ----
with open("model/heart_mlp_model.pkl", "rb") as f:
    md = pickle.load(f)

weights = [np.array(w) for w in md["weights"]]
biases  = [np.array(b) for b in md["biases"]]
X_mean  = np.array(md["X_mean"]).reshape(1, -1)
X_std   = np.array(md["X_std"]).reshape(1, -1)

N_FEATURES = X_mean.shape[1]


class PredictRequest(BaseModel):
    features: list[float]  # simple list; we'll validate length inside endpoint


@app.get("/health")
def health():
    return {"status": "ok", "n_features": N_FEATURES}


@app.post("/predict")
def predict(req: PredictRequest):
    if not isinstance(req.features, list):
        raise HTTPException(status_code=400, detail="features must be a list of numbers")
    if len(req.features) != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"features must contain exactly {N_FEATURES} values (got {len(req.features)})"
        )

    try:
        x = np.array(req.features, dtype=float).reshape(1, -1)
    except Exception:
        raise HTTPException(status_code=400, detail="features must be numeric")

    # normalize using training mean/std
    x = (x - X_mean) / X_std

    # forward pass
    out = forward_pass(x, weights, biases)
    prob = float(out.ravel()[0])
    label = int(prob > 0.5)

    return {"prediction": label, "probability": prob}
