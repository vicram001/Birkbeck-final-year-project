from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
import os

# replace with actual filename (without .py)
from filename import TextClassifier

PREDICTIONS_FILE = "saved_predictions.json"

# load model at startup
clf = TextClassifier("textcat_multilabel_model")

# load predictions from file (if it exists)
if os.path.exists(PREDICTIONS_FILE):
    with open(PREDICTIONS_FILE, "r") as f:
        saved_predictions = json.load(f)
else:
    saved_predictions = []

# request schema
class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = None

# response schema
class PredictResponse(BaseModel):
    text: str
    predicted_labels: List[str]
    all_scores: dict

app = FastAPI(title="TextClassifier API")

def save_predictions():
    """Write predictions to disk in JSON format."""
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(saved_predictions, f, indent=2)

# input endpoint
@app.post("/predict/", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = clf.predict(request.text, top_k=request.top_k, return_scores=True)
    saved_predictions.append(result)
    save_predictions()  # persist after each prediction
    return {
        "text": result["text"],
        "predicted_labels": result["predicted_labels"],
        "all_scores": result["all_scores"]
    }

# return endpoint
@app.get("/predictions/")
def list_predictions():
    return saved_predictions

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
