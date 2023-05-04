import os 
import sys 

import numpy as np
import pandas as pd 
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_obj(file_path,obj):
    try:
        
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_moedls(x_train, y_train, x_test, y_test, models):
    try:
        report={}
        
        for i in range(len(list(models))):
            model=list(models.values())[i]
            
            logging.info(f'x features:{x_train.shape},y features:{y_train.shape},model using:{list(models.keys())[i]}')
            try:
                model.fit(x_train, y_train)
            except:
                logging.info(f"error occured : {list(models.keys())[i]}")
            y_train_predict=model.predict(x_train)
            
            y_test_predict=model.predict(x_test)
            
            train_model_score=r2_score(y_train,y_train_predict)
            
            test_model_score=r2_score(y_test,y_test_predict)
            
            report[list(models.keys())[i]]=[test_model_score,train_model_score]
            
        return report
    
    except Exception as e:
        logging.info("Error has occured in evaluate model(in utils)")
        raise CustomException(e,sys)