"""Tests for FastAPI moderation endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import model_manager


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_without_model(client):
    model_manager._model = None
    response = client.post("/predict", json={"text": "Hello world"})
    assert response.status_code == 503
