import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts', 'train_data.csv')
    test_data_path : str = os.path.join('artifacts', 'test_data.csv')
    raw_data_path : str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion initiated by method(intiate_data_ingestion) of DataIngestion class')

        try:
            data_path = os.path.join('notebooks', 'data', 'gemstone.csv')
            df = pd.read_csv(data_path)
            logging.info('Data read as pandas DataFrame for furthur ingestion step ...')

            artifacts_folder_path = os.path.dirname(self.ingestion_config.raw_data_path)
            os.makedirs(artifacts_folder_path, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train Test split of raw data even before EDA and preprocessing steps applied on raw data')
            train_df, test_df = train_test_split(df, train_size=0.3, random_state=42, shuffle=True)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion finished ...')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error('Something went wrong during Data Ingestion process !!!')
            raise CustomException(e, sys)


