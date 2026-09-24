"""Unit tests for FastAPI REST API endpoints."""

import io
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["supported_classes_count"] == 8
    assert "predict" in data["endpoints"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "whitevision-api"


def test_classes_endpoint():
    response = client.get("/classes")
    assert response.status_code == 200
    classes = response.json()
    assert len(classes) == 8
    class_names = [c["class_name"] for c in classes]
    assert "bulldozer" in class_names
    assert "excavator" in class_names
    assert "dump_truck" in class_names
    for item in classes:
        assert len(item["code"]) == 3
        assert item["code"].isupper()


def test_predict_endpoint_valid_image(tmp_path: Path):
    # Create synthetic test image
    img = Image.new("RGB", (300, 300), color=(180, 160, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"image": ("test.jpg", buf, "image/jpeg")},
        params={"top_k": 3, "threshold": 0.10}
    )
    # The dummy image may either pass threshold or be rejected depending on domain gatekeeper
    assert response.status_code in [200, 422]
    if response.status_code == 200:
        data = response.json()
        assert data["is_supported"] is True
        assert "prediction" in data
        assert "confidence" in data
        assert len(data["top_predictions"]) == 3
        assert len(data["all_probabilities"]) == 8


def test_predict_endpoint_unsupported_image():
    ferrari_path = Path("scratch/ferrari_clean.jpg")
    if not ferrari_path.exists():
        pytest.skip("Ferrari test image not present")

    with open(ferrari_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"image": ("ferrari.jpg", f, "image/jpeg")},
            params={"top_k": 3}
        )
    assert response.status_code == 422
    assert "sports car" in response.json()["detail"].lower() or "passenger" in response.json()["detail"].lower()
