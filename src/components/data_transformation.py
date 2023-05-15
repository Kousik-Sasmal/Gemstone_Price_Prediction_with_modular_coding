import os
import sys

sys.path.append('/home/kousik/Desktop/PW_project/Diamond_price_prediction')
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated.')

            # separate the numerical and categorical columns
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_cols = ['cut', 'color', 'clarity']

            # rank the ordinal categorical
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            num_pipe = Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            cat_pipe = Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipe,numerical_cols),
                ('cat_pipeline',cat_pipe,categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info('Exception occurred in Data Transformation Process.')
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train data and Test data read successfully')

            target_column = 'price'
            drop_column = [target_column,'id']

            logging.info('Dividing the dataset into dependent and independent features')
            # train data
            input_feature_train_df = train_df.drop(columns=drop_column)
            target_feature_train_df = train_df[target_column]

            # test data
            input_feature_test_df = test_df.drop(columns=drop_column)
            target_feature_test_df = test_df[target_column]

            logging.info('Data Transformation starts')
            preprocesssor_obj = self.get_data_transformation_object()

            input_feature_train_arr = preprocesssor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocesssor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(obj=preprocesssor_obj,
                        file_path=self.data_transformation_config.preprocessor_obj_path)   

            logging.info('Data Transformation is completed')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            ) 

        except Exception as e:
            logging.info('Exception occurred in Data Transformation Process.')
            raise CustomException(e,sys)
