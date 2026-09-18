"""Unit tests for prediction logic and confidence thresholding."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from src.inference.predict import VehiclePredictor


def test_prediction_thresholding(tmp_path: Path):
    # Create a blank dummy image
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    img_path = tmp_path / "test_truck.jpg"
    img.save(img_path)

    # Instantiate predictor with high threshold
    predictor = VehiclePredictor(confidence_threshold=0.999)
    
    # If trained model is not present, mock predictor._model with a dummy predictor
    class DummyModel:
        def predict(self, batch, verbose=0):
            # Return uniform probabilities summing to 1.0 (each 0.125 < 0.999)
            return np.ones((len(batch), 8), dtype=np.float32) / 8.0
            
    if not predictor.model_path.exists():
        predictor._model = DummyModel()

    res = predictor.predict(img_path, top_k=3, threshold=0.999)

    assert "prediction" in res
    assert "confidence" in res
    assert "top_predictions" in res
    assert len(res["top_predictions"]) == 3
    assert "all_probabilities" in res
    assert len(res["all_probabilities"]) == 8

    # If confidence is below 0.999, prediction should be unknown
    if res["confidence"] < 0.999:
        assert res["prediction"] == "unknown"
        assert res["display_name"] == "Unknown / Low Confidence"
        assert not res["is_confident"]


def test_out_of_distribution_rejection():
    car_path = Path("scratch/ferrari_clean.jpg")
    if not car_path.exists():
        pytest.skip("Ferrari test crop not found in scratch directory")

    predictor = VehiclePredictor()
    res = predictor.predict(car_path, check_domain=True)

    assert not res["is_supported"], "Passenger car must be rejected by domain gatekeeper"
    assert res["prediction"] == "unsupported"
    assert "Rejected" in res["status"]
    assert res["error"] is not None
    assert "sports car" in res["error"].lower() or "passenger" in res["error"].lower()
    assert res["top_predictions"] == []

