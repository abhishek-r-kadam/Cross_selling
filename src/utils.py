import os
import sys
import pickle

from src.exception import CustomException

def save_object(file_path, obj):
    
    dir_path = os.path.dirname(file_path)

    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)

def load_object(file_path):
    
    with open(file_path, "rb") as file_obj:
        model = pickle.load(file_obj)
        return model
