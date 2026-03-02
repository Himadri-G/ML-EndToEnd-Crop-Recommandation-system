from pathlib import Path

from crop_yield_prediction.components.data_validation import DataValidation
from crop_yield_prediction.configuration.config import ConfigManager
from crop_yield_prediction.utils.logger import get_logger


logger = get_logger(
    name = __name__,
    file_name = "Pipeline.log"
    
)

STAGE_NAME = "Data Validation Stage"


def main():
    logger.info(f"{STAGE_NAME} is started...")
    
    config_manager = ConfigManager(
        file_path = Path("config/config.yaml")
    )
    
    data_validation_config = config_manager.get_data_config()
    
    data_validation = DataValidation(config = data_validation_config)
    
    status = data_validation.main_DataValidation_part()
    
    if not status:
        raise Exception(f"{STAGE_NAME} is failed")
    
    logger.info(f"{STAGE_NAME} is completed !")
    
if __name__ == "__main__":
    main()