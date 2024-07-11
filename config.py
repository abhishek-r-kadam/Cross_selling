import os
import sys




class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_dir = "artifacts"
        self.raw_data_path = os.path.join(self.data_ingestion_dir,"raw_data.csv")
        self.train_data_path = os.path.join(self.data_ingestion_dir,"train_data.csv")
        self.test_data_path = os.path.join(self.data_ingestion_dir,"test_data.csv")
        
        
class DataTranformationConfig:
    def __init__(self):
        self.preprocess_dir = "artifacts"
        self.preprocessor_path = os.path.join(self.preprocess_dir,"preprocessor.pkl")
        
        self.num_col = [
            'Age',
            'Driving_License',
            'Region_Code',
            'Previously_Insured',
            'Annual_Premium',
            'Policy_Sales_Channel',
            'Vintage'
        ]
        
        self.cat_col = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
        
        self.target_col = "Response"
        
class ModelTrainingConfig:
    def __init__(self):
        self.model_dir = "artifacts"
        self.model_path = os.path.join(self.model_dir,"model.pkl")
        