from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# replace with actual filename (without .py)
from filename import TextClassifier  

# load model at startup
clf = TextClassifier("textcat_multilabel_model")

# store predictions in memory
saved_predictions = []

# request schema
class PredictRequest(BaseModel):
    text: str
    top_k: int | None = None

# response schema
class PredictResponse(BaseModel):
    text: str
    predicted_labels: List[str]
    all_scores: dict

app = FastAPI(title="TextClassifier API")

# input endpoint
@app.post("/predict/", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = clf.predict(request.text, top_k=request.top_k, return_scores=True)
    saved_predictions.append(result)  # Save for history
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
