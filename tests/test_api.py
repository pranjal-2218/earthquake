import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="module")
def client():
    """Fixture to provide a test client with app lifespan context managed."""
    with TestClient(app) as c:
        yield c

def test_read_main(client):
    """Test that health check endpoint returns 200 and online status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "online",
        "message": "SeismoCast API is running and ready for inferences."
    }

def test_model_info(client):
    """Test that model-info endpoint returns metadata and metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_metadata" in data
    assert "metrics" in data
    assert "classifier" in data["metrics"]
    assert "regressor" in data["metrics"]

def test_prediction_valid(client):
    """Test prediction endpoint with valid payload."""
    payload = {
        "latitude": 20.0,
        "longitude": 78.0,
        "depth": 10.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "earthquake_prone" in data
    assert "predicted_magnitude" in data
    assert isinstance(data["earthquake_prone"], int)
    assert isinstance(data["predicted_magnitude"], float)

def test_prediction_invalid_latitude(client):
    """Test that latitude out of bounds ge=-90.0, le=90.0 returns 422 Validation Error."""
    payload = {
        "latitude": 120.0,
        "longitude": 78.0,
        "depth": 10.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_prediction_invalid_longitude(client):
    """Test that longitude out of bounds ge=-180.0, le=180.0 returns 422 Validation Error."""
    payload = {
        "latitude": 20.0,
        "longitude": -200.0,
        "depth": 10.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_prediction_invalid_depth(client):
    """Test that negative depth returns 422 Validation Error."""
    payload = {
        "latitude": 20.0,
        "longitude": 78.0,
        "depth": -5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_prediction_with_optional_fields(client):
    """Test prediction endpoint with all optional parameters provided."""
    payload = {
        "latitude": 20.0,
        "longitude": 78.0,
        "depth": 10.0,
        "nst": 15.0,
        "tsunami": 0,
        "rms": 0.2,
        "gap": 110.0,
        "dmin": 0.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "earthquake_prone" in data
    assert "predicted_magnitude" in data
    assert isinstance(data["earthquake_prone"], int)
    assert isinstance(data["predicted_magnitude"], float)
