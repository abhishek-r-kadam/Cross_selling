import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import numpy as np
import pandas as pd

import pandas as pd

class CustomData:
    def __init__(self, Gender	,Age	,Driving_License,	Region_Code	,Previously_Insured	,Vehicle_Age,	Vehicle_Damage,	Annual_Premium	,Policy_Sales_Channel,	Vintage):
                self.Gender = Gender
                self.Age = Age
                self.Driving_License = Driving_License
                self.Region_Code = Region_Code
                self.Previously_Insured = Previously_Insured
                self.Vehicle_Age = Vehicle_Age
                self.Vehicle_Damage = Vehicle_Damage
                self.Annual_Premium = Annual_Premium
                self.Policy_Sales_Channel = Policy_Sales_Channel
                self.Vintage = Vintage
                
    def get_data_as_data_frame(self):
        data = {
            'Gender': [self.Gender],
            "Age":[self.Age],
            'Driving_License': [self.Driving_License],
            'Region_Code': [self.Region_Code],
            'Previously_Insured': [self.Previously_Insured],
            'Vehicle_Age': [self.Vehicle_Age],
            'Vehicle_Damage': [self.Vehicle_Damage],
            'Annual_Premium': [self.Annual_Premium],
            'Policy_Sales_Channel': [self.Policy_Sales_Channel],
            'Vintage': [self.Vintage]
        }
        return pd.DataFrame(data)
    
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            prediction = np.where(preds >= 0.5, 1, 0)
            
            return prediction.tolist()  # Convert numpy array to list
        
        except Exception as e:
            raise CustomException(e, sys)