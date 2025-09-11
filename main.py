from fastapi import FastAPI, HTTPException, Query, Depends, Header
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import json
import os
import httpx

from dotenv import load_dotenv
load_dotenv()

from classifier import TextClassifier

# --- simple API key guard (optional) ---
API_KEY = os.getenv("API_KEY", "")  # set in .env
async def require_api_key(x_api_key: str = Header(None)):
    if not API_KEY:  # disabled if empty
        return
    if x_api_key != API_KEY:
        raise HTTPException(401, "Unauthorized")

# --- config (.env with defaults) ---
MODEL_DIR = os.getenv("MODEL_DIR", "textcat_multilabel_model")
PREDICTIONS_FILE = os.getenv("PREDICTIONS_FILE", "saved_predictions.json")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = os.getenv("RELOAD", "true").lower() == "true"

# LLM config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

# --- load model + predictions ---
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

# --- schemas ---
class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = None

class PredictResponse(BaseModel):
    text: str
    predicted_labels: List[str]
    all_scores: Dict[str, float]

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 300

# --- App ---
app = FastAPI(title="TextClassifier API + LLM Gateway")

def save_predictions() -> None:
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(saved_predictions, f, indent=2, ensure_ascii=False)

# ---- Classifier endpoints ----

# input endpoint
@app.post("/predict/", response_model=PredictResponse)  # add dependencies=[Depends(require_api_key)] if required
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
@app.get("/predictions/")  # add dependencies=[Depends(require_api_key)] if required
def list_predictions():
    return saved_predictions

# clear results endpoint
@app.post("/clear_predictions/")  # add dependencies=[Depends(require_api_key)] if required
def clear_predictions():
    global saved_predictions
    saved_predictions = []
    save_predictions()
    return {"message": "All predictions cleared."}

# ---- LLM endpoint (external APIs) ----
@app.post("/chat/", dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest, provider: str = Query("openai", enum=["openai", "anthropic"])):
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            if provider == "openai":
                if not OPENAI_API_KEY:
                    raise HTTPException(500, "Missing OPENAI_API_KEY")
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENAI_MODEL,
                        "messages": [{"role": "user", "content": req.prompt}],
                        "max_tokens": req.max_tokens,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return {
                    "provider": "openai",
                    "model": OPENAI_MODEL,
                    "response": data["choices"][0]["message"]["content"],
                }

            # provider == "anthropic"
            if not ANTHROPIC_API_KEY:
                raise HTTPException(500, "Missing ANTHROPIC_API_KEY")
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": req.max_tokens,
                    "messages": [{"role": "user", "content": req.prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            return {
                "provider": "anthropic",
                "model": ANTHROPIC_MODEL,
                "response": "".join(text_blocks),
            }

    except httpx.HTTPStatusError as e:
        detail = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=e.response.status_code if e.response else 502,
                            detail=f"Upstream error: {detail}") from e
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Network error: {e}") from e

if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=RELOAD)
