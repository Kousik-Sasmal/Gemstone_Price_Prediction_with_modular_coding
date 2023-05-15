import os
import sys
import numpy as np
import pandas as pd

sys.path.append('/home/kousik/Desktop/PW_project/Diamond_price_prediction')
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            logging.info('Preprocessor and Model objects are loaded for Predict Pipeline.')

            transformed_data = preprocessor.transform(features)
            logging.info('New test data is transformed.')

            prediction = model.predict(transformed_data)
            logging.info('Prediction of New test data is done.')
            
            return prediction
        
        except Exception as e:
            logging.info('Error occurred during Prediction.')
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            customdata_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }

            logging.info(f'df dict is {customdata_input_dict}')
            df = pd.DataFrame(customdata_input_dict)
            logging.info('User Input gathered as Dataframe')
            return df

        except Exception as e:
            logging.info('Error occurred in Prediction Pipeline.')
            raise CustomException(e,sys)
