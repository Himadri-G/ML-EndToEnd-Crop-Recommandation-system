import logging
from pathlib import Path
import mlflow
from crop_yield_prediction.configuration.config import ConfigManager
from crop_yield_prediction.components.model_training import ModelTraining
from crop_yield_prediction.utils.logger import get_logger

STAGE_NAME = "Model Training Stage"

logger = get_logger(
    name=__name__,
    log_file="pipeline.log"
)

def main():
    logger.info(f"{STAGE_NAME} started.........")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Crop_Yield_Prediction_Training")

    config_manager = ConfigManager(Path("config/config.yaml"))
    training_config = config_manager.get_model_training_config()

    trainer = ModelTraining(config=training_config)
    trainer.main_model_training()

    logger.info(f"{STAGE_NAME} completed")

if __name__ == "__main__":
    main()