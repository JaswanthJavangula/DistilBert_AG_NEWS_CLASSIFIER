
"""FastAPI service for AG News topic classification."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_text

app = FastAPI(title="AG News Topic Classifier", version="1.0.0")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      description="News article text")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict


@app.get("/health")
def health():
    # TODO 1: return status and model name
    return {"status": "ok", "model": "distilbert-base-uncased-finetuned-ag-news"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # TODO 2: call predict_text, wrap failures in HTTPException(500)
    # hint: try/except, raise HTTPException(status_code=500, detail=str(e))
    try:
        return predict_text(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))