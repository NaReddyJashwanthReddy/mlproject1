# it consist the logging details
import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"   #create a filename with mmm,dd,YY,HH,MM,SS on it
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) # create a path
os.makedirs(logs_path,exist_ok=True) # create a directory in that path exist_ ok means even tough there is a file with that name append it anyways

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

#when ever we want to create a log file/ for overwriting the logconfig
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    
)

