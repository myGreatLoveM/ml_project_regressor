import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        logging.info('Prepartion begins for getting Data Transformation object ...')

        try:
            df = pd.read_csv(os.path.join('notebooks', 'data', 'gemstone.csv'))
            df = df.drop('id', axis=1)
            cat_cols = list(df.select_dtypes(exclude='number').columns)
            num_cols = list(df.select_dtypes(include='number').drop('price', axis=1).columns)

            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            categories = [cut_categories, color_categories, clarity_categories]

            logging.info('Pipeline(preprocessor) initiated ...')
            num_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordianlencoder', OrdinalEncoder(categories=categories)),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info(num_cols)
            logging.info(cat_cols)
            preprocessor = ColumnTransformer(
                transformers= [
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols),
                ]
            )
            logging.info('Pipeline complete')

            return preprocessor
        
        except Exception as e:
            logging.error('Something went wrong while preparing Data Transformation object (get_data_transformation_object) !!!')
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        logging.info('Data Transformation begins ...')

        try:
            train_data_df = pd.read_csv(train_data_path)
            test_data_df = pd.read_csv(test_data_path)
            
            logging.info('Read train and test data as pandas DataFrame')
            logging.info(f'Train DataFrame Head : \n {train_data_df.head().to_string()}')
            logging.info(f'Test DataFrame Head : \n {test_data_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            target_column_name = 'price'

            input_features_train_df = train_data_df.drop(['id', target_column_name], axis=1)
            target_feature_train_df = train_data_df[target_column_name]

            input_features_test_df = test_data_df.drop(['id', target_column_name], axis=1)
            target_feature_test_df = test_data_df[target_column_name]

            preprocessing_obj = self.get_data_transformation_object()

            logging.info(list(input_features_train_df.columns))
            logging.info(list(input_features_test_df.columns))

            logging.info('Applying preprocessing object on training and testing datasets ...') 

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_data_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_data_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            logging.info('Applied preprocessing on data ..')

            save_object(
                obj_file_path=self.data_transformation_config.preprocessor_obj_file_path, 
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_data_arr,
                test_data_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error('Something went wrong while transforming data')
            raise CustomException(e, sys)

