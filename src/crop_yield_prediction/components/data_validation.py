import os
import yaml

import pandas as pd

from crop_yield_prediction.entity.config_entity import DataValidationConfig
from crop_yield_prediction.utils.logger import get_logger

logger = get_logger(
    
    name = __name__,
    log_file = "data_validation.log"
    
)

class DataValidation:
    
    def __init__(self , config: DataValidationConfig):
        self.config = config
        
    def _read_schema(self):
        with open(self.config.schema_file , "r") as file:
            schema = yaml.safe_load(file)
        return schema
    
    def main_DataValidation_part(self):
        logger.info("Data Validation Started !")
        
        schema = self._read_schema()
        expected_columns = list(schema["columns"].keys())
        
        df = pd.read_csv(self.config.train_dir)
        actual_columns = list(df.columns)
            
        if len(expected_columns) != len(actual_columns):
            logger.error("Column count mismatched !")
            status = False

        elif set(expected_columns) != set(actual_columns):
            logger.error("Column names mismatch !")
            status = False
                
        else:
            logger.info("Data Validation Completed !")
            status = True
        
        os.makedirs(self.config.root_dir , exist_ok= True)
        with open(self.config.validation_status_file , "w") as f:
            f.write(str(status))
        return status