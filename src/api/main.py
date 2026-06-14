import os
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("seismocast_api")

# Global dict to store loaded models and medians
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle model and median loading during startup."""
    clf_path = "models/classifier.pkl"
    reg_path = "models/regressor.pkl"
    medians_path = "models/medians.pkl"
    
    logger.info("Loading ML models and features medians...")
    if os.path.exists(clf_path) and os.path.exists(reg_path) and os.path.exists(medians_path):
        try:
            ml_models["clf"] = joblib.load(clf_path)
            ml_models["reg"] = joblib.load(reg_path)
            ml_models["medians"] = joblib.load(medians_path)
            logger.info("Successfully loaded classifier, regressor, and feature medians.")
        except Exception as e:
            logger.error(f"Failed to load model pickle files: {str(e)}")
    else:
        logger.warning(
            "Model or median files not found. Please run the training script "
            "('python -m src.models.train_models') first."
        )
    yield
    # Clean up resources on shutdown
    logger.info("Cleaning up ML models...")
    ml_models.clear()

app = FastAPI(
    title="SeismoCast API",
    description="Production-grade API for predicting earthquake magnitude and classification risk based on location features.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
from typing import Optional

class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude coordinate", examples=[20.0])
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude coordinate", examples=[78.0])
    depth: float = Field(..., ge=0.0, le=1000.0, description="Earthquake depth in kilometers", examples=[10.0])
    
    # Optional physical parameters. If omitted, model training medians will be used.
    nst: Optional[float] = Field(None, ge=0.0, description="Number of reporting stations")
    tsunami: Optional[int] = Field(None, ge=0, le=1, description="Tsunami warning generated (1 = Yes, 0 = No)")
    rms: Optional[float] = Field(None, ge=0.0, description="Root mean square travel time residual")
    gap: Optional[float] = Field(None, ge=0.0, le=360.0, description="Azimuthal gap in degrees")
    dmin: Optional[float] = Field(None, ge=0.0, description="Minimum distance to station in degrees")

class PredictResponse(BaseModel):
    earthquake_prone: int = Field(..., description="Risk class: 1 if high magnitude (>5.0) risk, 0 otherwise")
    predicted_magnitude: float = Field(..., description="Continuous regression prediction of earthquake magnitude")

@app.get("/", tags=["General"])
def home():
    """Returns basic health check message."""
    return {
        "status": "online",
        "message": "SeismoCast API is running and ready for inferences."
    }

@app.get("/model-info", tags=["Metrics"])
def get_model_info():
    """Retrieves current model evaluation metrics generated during training."""
    metrics_path = "models/metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics report not found. Has the model been trained?"
        )
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return {
            "model_metadata": {
                "algorithm": "RandomForest (Classifier & Regressor)",
                "features": ["latitude", "longitude", "depth", "nst", "tsunami", "rms", "gap", "dmin"]
            },
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error reading metrics file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read model metrics."
        )

@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"]
)
def predict(payload: PredictRequest):
    """
    Accepts latitude, longitude, depth, and optional physical parameters to predict 
    earthquake risk class (prone vs not prone) and predicted magnitude using trained RandomForest models.
    """
    # Check if models are loaded
    if "clf" not in ml_models or "reg" not in ml_models or "medians" not in ml_models:
        logger.error("Inference requested, but models or medians are not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models or medians are not loaded. Try running model training."
        )

    try:
        # Impute missing values with loaded training medians if they are not provided
        medians = ml_models["medians"]
        nst_val = payload.nst if payload.nst is not None else medians["nst"]
        tsunami_val = payload.tsunami if payload.tsunami is not None else medians["tsunami"]
        rms_val = payload.rms if payload.rms is not None else medians["rms"]
        gap_val = payload.gap if payload.gap is not None else medians["gap"]
        dmin_val = payload.dmin if payload.dmin is not None else medians["dmin"]

        # Construct input DataFrame for models in correct feature order
        features_order = ["latitude", "longitude", "depth", "nst", "tsunami", "rms", "gap", "dmin"]
        data = pd.DataFrame(
            [[
                payload.latitude, 
                payload.longitude, 
                payload.depth,
                nst_val,
                tsunami_val,
                rms_val,
                gap_val,
                dmin_val
            ]],
            columns=features_order
        )

        # Make predictions
        prone = ml_models["clf"].predict(data)[0]
        magnitude = ml_models["reg"].predict(data)[0]

        logger.info(
            f"Prediction made successfully for lat={payload.latitude}, "
            f"lon={payload.longitude}, depth={payload.depth} -> "
            f"prone={prone}, magnitude={magnitude:.2f}"
        )

        return PredictResponse(
            earthquake_prone=int(prone),
            predicted_magnitude=float(magnitude)
        )
    except Exception as e:
        logger.error(f"Error during model prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while running predictions."
        )