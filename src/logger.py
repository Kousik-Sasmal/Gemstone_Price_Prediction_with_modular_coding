import logging
import os
from datetime import datetime

dir_path = os.path.join(os.getcwd(),"logs")

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" 

file_path = os.path.join(dir_path,LOG_FILE)
os.makedirs(file_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(file_path,LOG_FILE)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= "[ %(asctime)s ] - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)