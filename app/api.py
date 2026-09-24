"""FastAPI Inference Service for Construction-Site Vehicle Classification.

Engineered for precision inference, zero-emoji telemetry, and cloud deployment on Render.
"""

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import VehiclePredictor
from src.dataset.normalization import get_normalizer

app = FastAPI(
    title="WhiteVision Vehicle Intelligence API",
    description="High-performance neural REST API for detecting and classifying construction equipment from images using EfficientNetB0.",
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
_normalizer = None


def get_normalizer_instance():
    global _normalizer
    if _normalizer is None:
        _normalizer = get_normalizer("config/classes.yaml")
    return _normalizer


def get_predictor() -> VehiclePredictor:
    global _predictor
    if _predictor is None:
        _predictor = VehiclePredictor()
    return _predictor


class TopPredictionItem(BaseModel):
    rank: Optional[int] = Field(None, description="Ranking position (1-based)")
    cls: str = Field(..., alias="class", description="Predicted vehicle class identifier")
    code: Optional[str] = Field(None, description="3-letter uppercase vehicle code, e.g. [BDZ], [EXC]")
    display_name: Optional[str] = Field(None, description="Human readable display name")
    confidence: float = Field(..., description="Probability confidence score (0.0 - 1.0)")
    percentage: Optional[str] = Field(None, description="Formatted percentage string")


class PredictionResponse(BaseModel):
    is_supported: bool = Field(True, description="Whether the image is within the construction vehicle domain")
    prediction: str = Field(..., description="Top predicted canonical vehicle class or 'unknown'")
    code: str = Field(..., description="3-letter vehicle code or UNK")
    display_name: str = Field(..., description="Display name of primary prediction")
    confidence: float = Field(..., description="Confidence of the primary prediction")
    percentage: str = Field(..., description="Formatted percentage string")
    status: str = Field(..., description="Classification status description")
    is_confident: bool = Field(..., description="Whether confidence meets threshold")
    threshold: float = Field(..., description="Cutoff threshold applied")
    top_predictions: List[TopPredictionItem] = Field(..., description="Ranked list of candidate predictions")
    all_probabilities: Dict[str, float] = Field(..., description="Probability distribution across all 8 classes")


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Service health indicator")
    service: str = Field("whitevision-api", description="Service identifier")
    version: str = Field("1.0.0", description="API version")


class ClassItem(BaseModel):
    class_name: str
    code: str
    display_name: str
    color: str


@app.get("/", tags=["Root"])
async def root_info():
    """Root endpoint providing service overview and telemetry links."""
    return {
        "service": "WhiteVision Vehicle Intelligence API",
        "version": "1.0.0",
        "status": "online",
        "architecture": "EfficientNetB0 (Transfer Learning) + Aspect-Preserving TTA",
        "supported_classes_count": 8,
        "endpoints": {
            "health": "/health",
            "classes": "/classes",
            "predict": "/predict",
            "documentation": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Health check endpoint to verify service availability."""
    return HealthResponse(status="healthy", service="whitevision-api", version="1.0.0")


@app.get("/classes", response_model=List[ClassItem], tags=["Metadata"])
async def list_classes():
    """Retrieve all 8 canonical construction vehicle categories with ASCII codes and colors."""
    normalizer = get_normalizer_instance()
    classes = normalizer.get_classes()
    return [
        ClassItem(
            class_name=c,
            code=normalizer.get_code(c),
            display_name=normalizer.get_display_name(c),
            color=normalizer.get_color(c),
        )
        for c in classes
    ]


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_vehicle(
    image: UploadFile = File(..., description="Image file (JPG, PNG, JPEG) of construction vehicle"),
    top_k: int = Query(3, ge=1, le=8, description="Number of top predictions to include"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Optional confidence threshold override"),
    use_tta: bool = Query(True, description="Enable multi-scale aspect-preserved test-time augmentation"),
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
        res = predictor.predict(pil_img, top_k=top_k, threshold=threshold, use_tta=use_tta)

        if not res.get("is_supported", True):
            raise HTTPException(
                status_code=422,
                detail=res.get("error", "Unsupported image: No construction-site vehicle detected.")
            )

        top_list = [
            TopPredictionItem(
                rank=item.get("rank"),
                **{
                    "class": item["class"],
                    "code": item.get("code"),
                    "display_name": item.get("display_name"),
                    "confidence": round(item["confidence"], 4),
                    "percentage": item.get("percentage"),
                }
            )
            for item in res["top_predictions"]
        ]

        return PredictionResponse(
            is_supported=True,
            prediction=res["prediction"],
            code=res.get("code", "UNK"),
            display_name=res.get("display_name", ""),
            confidence=round(res["confidence"], 4),
            percentage=res.get("percentage", f"{res['confidence'] * 100:.2f}%"),
            status=res.get("status", "Identified"),
            is_confident=res.get("is_confident", True),
            threshold=res.get("threshold", 0.50),
            top_predictions=top_list,
            all_probabilities=res.get("all_probabilities", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
