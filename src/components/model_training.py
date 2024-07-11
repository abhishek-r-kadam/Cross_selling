import os
import sys

from src.exception import CustomException
from src.logger import logging
from config import ModelTrainingConfig

from src.utils import save_object

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

class ModelTraining:
    def __init__(self):
        self.model_train_config = ModelTrainingConfig()
        
        
    def initiate_model_training(self,train_arr,test_arr):       
        
        try:
            self.model = LinearRegression()
            
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            self.model.fit(X=X_train,y=y_train)
            
            self.y_train_pred = self.model.predict(X=X_train)
            self.y_test_pred = self.model.predict(X=X_test)
            
            y_train_pred = np.where(self.y_train_pred >= 0.5, 1, 0)
            y_test_pred = np.where(self.y_test_pred >= 0.5, 1, 0)
            
            save_object(self.model_train_config.model_path,self.model)
            
            train_accuracy = accuracy_score(y_train,y_train_pred)
            test_accuracy = accuracy_score(y_test,y_test_pred)
            
            train_cnf = confusion_matrix(y_train,y_train_pred)
            test_cnf = confusion_matrix(y_test,y_test_pred)
            
            train_clf_report = classification_report(y_train,y_train_pred)
            test_clf_report = classification_report(y_test,y_test_pred)
            
            print('Model performance for Training set')
            print('Accuracy: {:.2f}'.format(train_accuracy))
            print("Confusion Matrix")
            print(train_cnf)
            print(train_clf_report)
            

            print('----------------------------------')
            
            print('Model performance for Test set')
            print('Accuracy: {:.2f}'.format(test_accuracy))
            print("Confusion Matrix")
            print(test_cnf)
            print(test_clf_report)
            
        except Exception as e:
            raise CustomException(e,sys)
        