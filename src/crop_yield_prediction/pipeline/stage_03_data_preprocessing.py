import logging
from pathlib import Path
from crop_yield_prediction.configuration.config import ConfigManager
from crop_yield_prediction.components.data_preprocessing import DataPreprocessing
from crop_yield_prediction.utils.logger import get_logger

STAGE_NAME = "Data Preprocessing Stage"

logger = get_logger(
    name=__name__,
    log_file="pipeline.log"
)

def main():
    logger.info(f"{STAGE_NAME} started.........")

    config_manager = ConfigManager(Path("config/config.yaml"))
    preprocessing_config = config_manager.get_data_preprocessing_config()

    preprocessor = DataPreprocessing(config=preprocessing_config)
    preprocessor.main_data_preprocessing()  # dynamically splits features/target and scales data

    logger.info(f"{STAGE_NAME} completed")

if __name__ == "__main__":
    main()