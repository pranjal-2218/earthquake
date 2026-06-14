import os
import json
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, root_mean_squared_error, r2_score
)
from src.data.fetch_data import USGSDataFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("model_training")

def train() -> None:
    """
    Fetches raw USGS earthquake data, splits it into training and testing sets,
    trains a Random Forest Classifier and Regressor using both spatial and physical features,
    evaluates both models, saves metrics to JSON, and exports model pickle files.
    """
    logger.info("Initializing USGS Data Fetcher...")
    fetcher = USGSDataFetcher()
    
    try:
        df = fetcher.fetch()
        logger.info(f"Successfully fetched {len(df)} earthquake records.")
    except Exception as e:
        logger.error(f"Failed to fetch earthquake data: {str(e)}")
        raise e

    # Drop records with missing magnitude target
    df = df.dropna(subset=["magnitude"])

    # Features: spatial coordinates, depth, and physical measurements
    features = ["latitude", "longitude", "depth", "nst", "tsunami", "rms", "gap", "dmin"]
    X = df[features]
    
    # Classification: High risk if magnitude > 5
    y_class = (df["magnitude"] > 5).astype(int)
    
    # Regression: Continuous magnitude values
    y_reg = df["magnitude"]

    # Perform Train/Test Split
    logger.info("Splitting data into train and test sets (80/20)...")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # Calculate and save training medians for API imputation
    medians = X_train_c.median().to_dict()
    os.makedirs("models", exist_ok=True)
    joblib.dump(medians, "models/medians.pkl")
    logger.info(f"Saved feature median values for imputation: {medians}")

    # Impute missing values using training medians
    for col in features:
        X_train_c = X_train_c.copy()
        X_train_c[col] = X_train_c[col].fillna(medians[col])
        X_test_c = X_test_c.copy()
        X_test_c[col] = X_test_c[col].fillna(medians[col])

        X_train_r = X_train_r.copy()
        X_train_r[col] = X_train_r[col].fillna(medians[col])
        X_test_r = X_test_r.copy()
        X_test_r[col] = X_test_r[col].fillna(medians[col])

    # Train Classifier (high precision focus with balanced class weights)
    logger.info("Training RandomForest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_c, y_train_c)
    
    # Evaluate Classifier
    clf_preds = clf.predict(X_test_c)
    clf_accuracy = accuracy_score(y_test_c, clf_preds)
    clf_precision = precision_score(y_test_c, clf_preds, zero_division=0)
    clf_recall = recall_score(y_test_c, clf_preds, zero_division=0)
    clf_f1 = f1_score(y_test_c, clf_preds, zero_division=0)
    
    logger.info(f"Classifier Metrics: Accuracy={clf_accuracy:.4f}, Precision={clf_precision:.4f}, Recall={clf_recall:.4f}, F1-Score={clf_f1:.4f}")

    # Train Regressor
    logger.info("Training RandomForest Regressor...")
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_r, y_train_r)
    
    # Evaluate Regressor
    reg_preds = reg.predict(X_test_r)
    reg_mae = mean_absolute_error(y_test_r, reg_preds)
    reg_rmse = root_mean_squared_error(y_test_r, reg_preds)
    reg_r2 = r2_score(y_test_r, reg_preds)
    
    logger.info(f"Regressor Metrics: MAE={reg_mae:.4f}, RMSE={reg_rmse:.4f}, R2={reg_r2:.4f}")

    # Save models
    logger.info("Exporting models as joblib pickle files...")
    joblib.dump(clf, "models/classifier.pkl",compress = 3)
    joblib.dump(reg, "models/regressor.pkl" ,compress = 3)

    # Save evaluation metrics
    metrics = {
        "classifier": {
            "accuracy": clf_accuracy,
            "precision": clf_precision,
            "recall": clf_recall,
            "f1_score": clf_f1
        },
        "regressor": {
            "mae": reg_mae,
            "rmse": reg_rmse,
            "r2_score": reg_r2
        }
    }
    
    metrics_path = "models/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved evaluation metrics report to {metrics_path}")

if __name__ == "__main__":
    train()
