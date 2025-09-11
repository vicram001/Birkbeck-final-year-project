from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import json
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TextClassifier API")

# CORS so Reflex UI (port 3000) can call the API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "http://34.46.126.152:3000" ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from dotenv import load_dotenv
# load .env into process env
load_dotenv()

from classifier import TextClassifier

# configuration via .env (with safe defaults)
MODEL_DIR = os.path.abspath(os.getenv("MODEL_DIR", "textcat_multilabel_model"))
PREDICTIONS_FILE = os.getenv("PREDICTIONS_FILE", "saved_predictions.json")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = os.getenv("RELOAD", "true").lower() == "true"

# load model + persisted predictions
clf = TextClassifier(MODEL_DIR)

if os.path.exists(PREDICTIONS_FILE):
    try:
        with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
            saved_predictions = json.load(f)
            if not isinstance(saved_predictions, list):
                saved_predictions = []
    except Exception:
        saved_predictions = []
else:
    saved_predictions = []

# schemas
class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = None

class PredictResponse(BaseModel):
    text: str
    predicted_labels: List[str]
    all_scores: Dict[str, float]

# create the FastAPI application object, give it a display name ("TextClassifier API") for docs (Swagger UI, ReDoc, etc.)
app = FastAPI(title="TextClassifier API")

def save_predictions() -> None:
    """Write predictions to disk in JSON format."""
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(saved_predictions, f, indent=2, ensure_ascii=False)

# input endpoint
@app.post("/predict/", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = clf.predict(request.text, top_k=request.top_k, return_scores=True)
    saved_predictions.append(result)
    save_predictions()
    return {
        "text": result["text"],
        "predicted_labels": result["predicted_labels"],
        "all_scores": result["all_scores"],
    }

# return endpoint
@app.get("/predictions/")
def list_predictions():
    return saved_predictions

# clear results endpoint
@app.post("/clear_predictions/")
def clear_predictions():
    """Clear all stored predictions (memory + file)."""
    global saved_predictions
    saved_predictions = []
    save_predictions()
    return {"message": "All predictions cleared."}

if __name__ == "__main__":
    # run with env-configured host/port/reload
    # example CLI: `python app.py`
    uvicorn.run("app:app", host=HOST, port=PORT, reload=RELOAD)
