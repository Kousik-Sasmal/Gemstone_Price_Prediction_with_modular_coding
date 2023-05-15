import os
import sys
import numpy as np
import pandas as pd

sys.path.append('/home/kousik/Desktop/PW_project/Diamond_price_prediction')
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from sklearn.linear_model import LinearRegression,Lasso,ridge_regression,ElasticNet
from sklearn.ensemble import RandomForestRegressor

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'LinearRegression':LinearRegression(),
                'RandomForest':RandomForestRegressor(),
                'Lasso':Lasso(),
                'ElasticNet':ElasticNet()
            }

            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models)
            logging.info(f'Model Report: {model_report}')

            best_model_index =sorted(list(enumerate(list(model_report.values()))),
                                     key=lambda x: x[1], reverse=True)[0][0]
            
            #best_model_score = max(model_report.values())

            best_model_name:str = list(model_report.keys())[best_model_index]
            best_model_score = list(model_report.values())[best_model_index]

            best_model = models[best_model_name]

            logging.info(f'Best Model is {best_model_name} and score is {best_model_score}')
            print(f'Best Model is {best_model_name} and score is {best_model_score}')

            save_object(
                obj=best_model,
                file_path=self.model_trainer_config.trained_model_path
            )
        
        except Exception as e:
            logging.info('Error occurred in model training')
            raise CustomException(e,sys)