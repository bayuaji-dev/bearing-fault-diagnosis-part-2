
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.predictor import prediksi_baris, muat_artifact

app = FastAPI(
    title="Bearing Fault Diagnosis API",
    description="API klasifikasi kondisi bearing menjadi normal, outer_race, inner_race, atau bola.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    data: dict[str, Any]

class BatchPredictionRequest(BaseModel):
    data: list[dict[str, Any]]

@app.get("/health")
def health():
    try:
        _, metadata = muat_artifact()
        return {"status": "ok", "versi_model": metadata.get("versi_model")}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return prediksi_baris(request.data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    try:
        return prediksi_baris(request.data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
