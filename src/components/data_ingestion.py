import os
import sys

sys.path.append('/home/kousik/Desktop/PW_project/Diamond_price_prediction')
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","data.csv")
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        try:
            df = pd.read_csv('notebooks/data/train.csv')
            logging.info("Data read successfull.")

            # creating the file path, if not created earlier
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Train-Test split initiated.")
            df_train,df_test = train_test_split(df,test_size=0.2,random_state=42)

            df_train.to_csv(self.ingestion_config.train_data_path,index=False)
            df_test.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Data Ingestion Process is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Error Occurred during Data Ingestion Process.")
            raise CustomException(e,sys)
