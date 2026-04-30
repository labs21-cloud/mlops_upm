import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_generate_valid_labels_returns_200():
    payload = {
        "labels": [0, 1, 2],
        "seed": 42
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "labels" in data
    assert "images_base64" in data
    assert len(data["labels"]) == 3
    assert len(data["images_base64"]) == 3
    assert all(len(img) > 100 for img in data["images_base64"])


def test_generate_invalid_labels_returns_400():
    payload = {
        "labels": [-1, 10],
        "seed": 42
    }
    response = client.post("/generate", json=payload)
    assert response.status_code in (400, 422)
    data = response.json()
    assert "detail" in data


def test_generate_empty_labels_returns_422():
    payload = {
        "labels": [],
        "seed": 42
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 422