import logging
from pathlib import Path
from crop_yield_prediction.configuration.config import ConfigManager
from crop_yield_prediction.components.data_validation import DataValidation
from crop_yield_prediction.utils.logger import get_logger

STAGE_NAME = "Data Validation Stage"

logger = get_logger(
    name=__name__,
    log_file="pipeline.log"
)

def main():
    logger.info(f"{STAGE_NAME} started.........")

    config_manager = ConfigManager(Path("config/config.yaml"))
    validation_config = config_manager.get_data_validation_config()

    validator = DataValidation(config=validation_config)
    status = validator.main_DataValidation_part()
    if not status:
        raise Exception(f"{STAGE_NAME} failed")

    logger.info(f"{STAGE_NAME} completed successfully")

if __name__ == "__main__":
    main()