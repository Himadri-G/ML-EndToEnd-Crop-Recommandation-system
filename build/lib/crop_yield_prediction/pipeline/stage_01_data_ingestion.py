import logging
from pathlib import Path
from crop_yield_prediction.configuration.config import ConfigManager
from crop_yield_prediction.components.data_ingestion import DataIngestion
from crop_yield_prediction.utils.logger import get_logger

STAGE_NAME = "Data Ingestion Stage"

logger = get_logger(
    name=__name__,
    log_file="pipeline.log"
)

def main():
    logger.info(f"{STAGE_NAME} started.........")

    config_manager = ConfigManager(Path("config/config.yaml"))
    ingestion_config = config_manager.get_data_ingestion_config()

    ingestion = DataIngestion(config=ingestion_config)
    ingestion.main_DataIngestion_part()

    logger.info(f"{STAGE_NAME} completed")

if __name__ == "__main__":
    main()