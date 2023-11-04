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
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    logging.info('Evaluating all models passed .. ')
    try:
        report = {}
        max_test_score = 0
        best_model_name = None

        for model in models.keys():

            curr_model = models[model]
            curr_model.fit(X_train, y_train)

            y_test_pred = curr_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            if (test_model_score > max_test_score):
                max_test_score = test_model_score
                best_model_name = model

            report[model] = test_model_score
        
        return (
            report,
            max_test_score,
            best_model_name
        )

    except Exception as e:
        logging.error('Something went wrong while evaluating the model')
        raise CustomException(e, sys)
