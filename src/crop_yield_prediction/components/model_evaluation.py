import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from crop_yield_prediction.entity.config_entity import ModelEvaluationConfig
from crop_yield_prediction.utils.logger import get_logger

logger = get_logger(__name__, "model_evaluation.log")


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def main_ModelEvaluation_part(self):
        """
        Main method to evaluate the trained model.
        Calculates R2, RMSE, MAE metrics, saves them locally, and logs to MLflow.
        """
        logger.info("Starting model evaluation for Crop Yield Prediction")

        # ------------------ LOAD TRAINED MODEL ------------------ #
        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(f"Model file not found at {self.config.model_path}")
        model = joblib.load(self.config.model_path)

        # ------------------ LOAD TEST DATA ------------------ #
        if not Path(self.config.preprocessed_test_dir).exists():
            raise FileNotFoundError(f"Test CSV file not found at {self.config.preprocessed_test_dir}")
        test_df = pd.read_csv(self.config.preprocessed_test_dir)

        if self.config.target_column not in test_df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in test CSV")
        
        X_test = test_df.drop(columns=[self.config.target_column])
        y_test = test_df[self.config.target_column]

        # ------------------ PREDICTIONS & METRICS ------------------ #
        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        metrics = {
            "r2_score": r2,
            "rmse": rmse,
            "mae": mae
        }

        # ------------------ SAVE METRICS LOCALLY ------------------ #
        Path(self.config.metrics_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved locally at {self.config.metrics_file}")

        # ------------------ LOG METRICS TO MLFLOW ------------------ #
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri or "sqlite:///mlflow.db")
        mlflow.set_experiment(self.config.mlflow_experiment_name or "Crop_Yield_Prediction_Evaluation")

        with mlflow.start_run(run_name="Model_Evaluation"):
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="evaluated_model")
        logger.info("Metrics logged to MLflow successfully")

        logger.info(f"Evaluation metrics: {metrics}")
        logger.info("Model evaluation completed successfully")
        return metrics