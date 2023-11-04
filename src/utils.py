import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(obj_file_path, obj):
    try:
        dir_path = os.path.dirname(obj_file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(obj_file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

