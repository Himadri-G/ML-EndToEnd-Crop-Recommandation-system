import os
import logging 
import pathlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from crop_yield_prediction.entity.config_entity import DataIngestionConfig
from crop_yield_prediction.utils.logger import get_logger


logger = get_logger(
    
    name = __name__,
    log_file= "data_ingestion.log"
    
)

class DataIngestion:
    
    def __init__(self,config : DataIngestionConfig):
        self.config = config
        
    def _read_data(self):
        df = pd.read_csv(self.config.source_dir)
        return df
    
    def _save_rawData(self , df):
        raw_data_path = Path(self.config.root_dir)/"raw.csv"
        df.to_csv(raw_data_path , index = False)
        return raw_data_path
    
    def _split_data(self, df):
        return train_test_split(df , test_size = 0.3, random_state= 42)
    
    def mainDataIngestionpart(self):
        logger.info("starting Data_ingestion")
        os.makedirs(self.config.root_dir , exist_ok= True)
        
        df = self._read_data()
        self._save_rawData(df)
        
        train_df , test_df = self._split_data(df)
        
        
        train_df.to_csv(self.config.train_dir , index = False)
        test_df.to_csv(self.config.test_dir , index = False)
        
        
        logger.info("Data Ingestion completed !") 
        
        
        return self.config.train_dir , self.config.test_dir      