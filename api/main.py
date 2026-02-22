from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from api.schemas import PredictionRequest, PredictionResponse
from src.models.cnn import load_model, predict

# holds the loaded model in memory
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once when the server starts — loads model into memory
    global model
    model = load_model("models/best_model.pt")
    print("Model loaded successfully")
    yield
    # runs when server shuts down
    print("Shutting down")


app = FastAPI(
    title="gRNA Efficiency Predictor",
    description="Predicts CRISPR Cas9 on-target efficiency from a 30-mer sequence",
    version="1.0.0",
    lifespan=lifespan
)


def get_category(score: float) -> str:
    if score >= 0.6:
        return "High"
    elif score >= 0.35:
        return "Medium"
    else:
        return "Low"


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict_efficiency(request: PredictionRequest):
    # validate sequence contains only valid nucleotides
    valid = set("ACGTacgt")
    if not all(c in valid for c in request.sequence):
        raise HTTPException(
            status_code=422,
            detail="Sequence contains invalid characters. Only A, C, G, T allowed."
        )

    score = predict(model, request.sequence)

    return PredictionResponse(
        sequence=request.sequence,
        efficiency_score=score,
        category=get_category(score),
        model_version="1.0.0"
    )
"""
---

**Step 7 — Create requirements.txt**

torch>=2.0.0
numpy>=1.24.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
"""