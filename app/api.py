"""FastAPI Inference Service for Construction-Site Vehicle Classification."""

import io
import sys
from pathlib import Path
from typing import List, Optional
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import VehiclePredictor

app = FastAPI(
    title="Construction Vehicle Classifier API",
    description="High-performance REST API for detecting and classifying construction equipment from images using EfficientNetB0.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor instance
_predictor: Optional[VehiclePredictor] = None


def get_predictor() -> VehiclePredictor:
    global _predictor
    if _predictor is None:
        _predictor = VehiclePredictor()
    return _predictor


class TopPredictionItem(BaseModel):
    cls: str = Field(..., alias="class", description="Predicted vehicle class identifier")
    confidence: float = Field(..., description="Probability confidence score (0.0 - 1.0)")


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Top predicted canonical vehicle class or 'unknown'")
    confidence: float = Field(..., description="Confidence of the primary prediction")
    top_predictions: List[TopPredictionItem] = Field(..., description="Ranked list of candidate predictions")


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Service health indicator")


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Health check endpoint to verify service availability."""
    return HealthResponse(status="healthy")


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_vehicle(
    image: UploadFile = File(..., description="Image file (JPG, PNG, JPEG) of construction vehicle"),
    top_k: int = Query(3, ge=1, le=8, description="Number of top predictions to include"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Optional confidence threshold override"),
):
    """Predict vehicle category from uploaded image file."""
    # Validate MIME type
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg", "application/octet-stream"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {image.content_type}. Please upload a JPEG or PNG image.",
        )

    # Read image contents safely
    contents = await image.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > 20 * 1024 * 1024:  # 20 MB security limit
        raise HTTPException(status_code=413, detail="Uploaded image exceeds 20MB limit.")

    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Corrupt or unreadable image file: {str(e)}")

    try:
        predictor = get_predictor()
        res = predictor.predict(pil_img, top_k=top_k, threshold=threshold)

        if not res.get("is_supported", True):
            raise HTTPException(
                status_code=422,
                detail=res.get("error", "Unsupported image: No construction-site vehicle detected.")
            )

        top_list = [
            TopPredictionItem(
                **{
                    "class": item["class"],
                    "confidence": round(item["confidence"], 4),
                }
            )
            for item in res["top_predictions"]
        ]

        return PredictionResponse(
            prediction=res["prediction"],
            confidence=round(res["confidence"], 4),
            top_predictions=top_list,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
