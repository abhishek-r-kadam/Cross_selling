import os
import sys

from src.exception import CustomException
from src.logger import logging

from config import DataIngestionConfig

import pandas as pd

from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTranformation
from src.components.model_training import ModelTraining

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
    train_path,test_path = data_ingestion.initiate_data_ingestion()
    
    datatransform = DataTranformation()
    train_arr,test_arr = datatransform.initiate_data_transformation(train_path=train_path,test_path=test_path)
    
    model_training = ModelTraining()
    model_training.initiate_model_training(train_arr=train_arr,test_arr=test_arr)
    
    
    

            



