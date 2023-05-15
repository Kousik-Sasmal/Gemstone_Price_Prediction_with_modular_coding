import os
import sys

sys.path.append('/home/kousik/Desktop/PW_project/Diamond_price_prediction')
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import numpy as np
import pandas as pd

from src.pipelines.prediction_pipeline import PredictPipeline

if __name__=="__main__":
    data_ingestion = DataIngestion()
    train_path,test_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array ,test_array, preprocessor_path = data_transformation.initiate_data_transformation(train_path,test_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_array,test_array)
