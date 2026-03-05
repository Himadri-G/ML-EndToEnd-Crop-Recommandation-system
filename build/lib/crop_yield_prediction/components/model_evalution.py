import json
import joblib
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error
)

from crop_yield_prediction.entity.config_entity import ModelEvaluationConfig
from crop_yield_prediction.utils.logger import get_logger


logger = get_logger(__name__, "model_evaluation.log")


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        logger.info("Starting model evaluation for Crop Yield Prediction")

        # Load trained model
        model = joblib.load(self.config.model_path)

        # Load test data
        test_df = pd.read_csv(self.config.processed_test_path)
        X_test = test_df.drop(columns=["yield"])
        y_test = test_df["yield"]

        # Predictions
        predictions = model.predict(X_test)

        # Regression Metrics
        r2 = r2_score(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions, squared=False)
        mae = mean_absolute_error(y_test, predictions)

        metrics = {
            "r2_score": r2,
            "rmse": rmse,
            "mae": mae
        }

        # Save metrics locally
        Path(self.config.metrics_file).parent.mkdir(
            parents=True, exist_ok=True
        )

        with open(self.config.metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        # MLflow tracking
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Crop_Yield_Prediction_Evaluation")

        with mlflow.start_run(run_name="Model_Evaluation"):
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="evaluated_model"
            )

        logger.info(f"Evaluation metrics: {metrics}")
        logger.info("Model evaluation completed successfully")

        return metrics