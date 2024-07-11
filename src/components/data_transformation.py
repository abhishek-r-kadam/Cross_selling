import os 
import sys
from typing import Tuple

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from config import DataTranformationConfig
from src.utils import save_object

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataTranformation:
    def __init__(self):
        self.data_tranforamtion_config = DataTranformationConfig()
        
    def preprocessor(self):
        try:
            num_col = self.data_tranforamtion_config.num_col
            cat_col = self.data_tranforamtion_config.cat_col
            
            num_pipeline = Pipeline(
                steps = [("scaler",StandardScaler())]
            )
            
            cat_pipeline = Pipeline(
                steps = [("one_hot_encoder",OneHotEncoder())]
            )
            
            preprocessor = ColumnTransformer(
                transformers = [
                    ("num_pipe",num_pipeline,num_col),
                    ("cat_pipe",cat_pipeline,cat_col)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Splitting data into X and y Started")
            X_train_df = train_df.drop(columns=[self.data_tranforamtion_config.target_col],axis=1)
            y_train_df = train_df[self.data_tranforamtion_config.target_col]
            
            X_test_df = test_df.drop(columns=[self.data_tranforamtion_config.target_col],axis=1)
            y_test_df = test_df[self.data_tranforamtion_config.target_col]
            logging.info("Completed")
            
            os.makedirs(os.path.join("artifacts","preprocessed"),exist_ok=True)
            
            preprocessor = self.preprocessor()
            logging.info("Processing Started.")
            X_train_df_processed = preprocessor.fit_transform(X_train_df)
            X_test_df_processed = preprocessor.transform(X_test_df)
            logging.info("Completed.")
            
            train_arr = np.c_[X_train_df_processed, np.array(y_train_df)]
            test_arr = np.c_[X_test_df_processed, np.array(y_test_df)]
            
            save_object(self.data_tranforamtion_config.preprocessor_path,preprocessor)
            logging.info("Object Saved")
            
            return (train_arr,test_arr)
            
        except Exception as e:
            raise CustomException(e,sys)
        
        