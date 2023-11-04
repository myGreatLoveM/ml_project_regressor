import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    data_ingestion_obj = DataIngestion()
    train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_data_arr, test_data_arr, _ = data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer_obj = ModelTrainer()
    model_trainer_obj.initiate_model_trainer(train_data_arr, test_data_arr)