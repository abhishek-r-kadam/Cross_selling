import os
import sys

from src.exception import CustomException
from src.logger import logging

from config import DataIngestionConfig

import pandas as pd

from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            logging.info("Starting Data Ingestion")
            df = pd.read_csv(r"data\train.csv")
            logging.info("Read Data Succesfully.")
            
            logging.info("Create Artifacts Directory To Save Data.")
            os.makedirs(os.path.join(self.data_ingestion_config.data_ingestion_dir),exist_ok=True)
            logging.info("file creates succesfully")
            
            logging.info("Started splitting Data.")
            train_dataset,test_dataset = train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Train Test Split Done.")
            
            logging.info("Saving data sets in Artifacts")
            df.to_csv(self.data_ingestion_config.raw_data_path)
            train_dataset.to_csv(self.data_ingestion_config.train_data_path)
            test_dataset.to_csv(self.data_ingestion_config.test_data_path)
            logging.info("Data sets saved")
            
            return (self.data_ingestion_config.train_data_path,self.data_ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()

            



