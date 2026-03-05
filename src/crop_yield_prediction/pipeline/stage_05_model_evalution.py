import logging
from pathlib import Path
from crop_yield_prediction.configuration.config import ConfigManager
from crop_yield_prediction.components.model_evalution import ModelEvaluation
from crop_yield_prediction.utils.logger import get_logger

STAGE_NAME = "Model Evaluation Stage"

logger = get_logger(
    name=__name__,
    log_file="pipeline.log"
)

def main():
    logger.info(f"{STAGE_NAME} started.........")

    config_manager = ConfigManager(Path("config/config.yaml"))
    evaluation_config = config_manager.get_model_evaluation_config()

    evaluator = ModelEvaluation(config=evaluation_config)
    evaluator.main_model_evaluation()

    logger.info(f"{STAGE_NAME} completed")

if __name__ == "__main__":
    main()