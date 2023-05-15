import os
import sys

sys.path.append('/home/kousik/Desktop/PW_project/Diamond_price_prediction')
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(obj,file_path):
    try:
        # creating the file path, if not created earlier
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        pickle.dump(obj,open(file_path,'wb'))

    except Exception as e:
        logging.info('Error occurred in saving the object.')
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        return pickle.load(open(file_path,'rb'))

    except Exception as e:
        logging.info('Error occurred in loading the object.')
        raise CustomException
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # model training
            model.fit(X_train,y_train)
            
            # prediction
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        logging.info('Model evaluation is completed.')

        return report

    except Exception as e:
        logging.info('Error in Model Training.')
        raise CustomException(e,sys)