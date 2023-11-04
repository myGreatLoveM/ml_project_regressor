import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

@dataclass
class ModelTrainerConfig:
    trained_model_obj_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_data_arr, test_data_arr):
        logging.info('Models training phase started ..')
        try:
            logging.info('Splitting Independent and dependent variables from train and test data')
            
            X_train, y_train, X_test, y_test = (
                train_data_arr[:, :-1],
                train_data_arr[:, -1],
                test_data_arr[:, :-1],
                test_data_arr[:, -1]
            )
        
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
            }

            models_report, best_model_score, best_model_name = evaluate_models(X_train, y_train, X_test, y_test, models)

            logging.info(f'Models report : {models_report}')
            logging.info(f'Best model {best_model_name} with score {best_model_score}')

            save_object(
                obj_file_path=self.model_trainer_config.trained_model_obj_file_path,
                obj=models[best_model_name]               
            )
            logging.info('Trained Model pkl file saved ...')

        except Exception as e:
            logging.error('Something went wrong during Model Training !!!')
            raise CustomException(e, sys)
