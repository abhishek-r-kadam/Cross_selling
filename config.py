import os
import sys

class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_dir = "artifacts"
        self.raw_data_path = os.path.join(self.data_ingestion_dir,"raw_data.csv")
        self.train_data_path = os.path.join(self.data_ingestion_dir,"train_data.csv")
        self.test_data_path = os.path.join(self.data_ingestion_dir,"test_data.csv")