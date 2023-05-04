import os 
import sys

from src.logger import logging 
from src.exception import CustomException
from src.utils import save_obj
from src.utils import evaluate_moedls

from dataclasses import dataclass 

from sklearn.linear_model import LinearRegression 
from catboost import CatBoostRegressor 
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.metrics import r2_score


@dataclass 
class ModelTrainingConfig:
    trained_model_file_path : str=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_traning_file_cofig=ModelTrainingConfig()
        
    def initilize_model_traniner(self,train_arr,test_arr):
        logging.info("startd the model for train array")
        try:
            logging.info("split training and testing data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            logging.info(f'number of features : {x_train.shape[1]},{y_train.shape}')
            models={
                "RandomForest":RandomForestRegressor(),
                "linearRegression":LinearRegression(),
                "GradientBoosting":GradientBoostingRegressor(),
                "CatBoosting":CatBoostRegressor(verbose=False),
                "XGBoost":XGBRegressor(),
                "adaBoost":AdaBoostRegressor(),
                "decisionTree":DecisionTreeRegressor(),
                "Kneighnour":KNeighborsRegressor()
            }
            
            model_report :dict=evaluate_moedls(x_train,y_train,x_test,y_test,models)
            
            # get best score from models
            best_model_score=max(model_report.values())
            
            # model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            if best_model_score[0] <= 0.6:
                logging.info(f'The best score is less than 0.6 so they performed poorly : {best_model_score}')
                raise CustomException("No best model found")
            logging.info(f'found the best model : {best_model_score}')    
            
            save_obj(
                file_path=self.model_traning_file_cofig.trained_model_file_path,
                obj=best_model
            )
            
            prediction=best_model.predict(x_test)
            
            r2_square=r2_score(y_test,prediction)
            
            return r2_square
            
            
        except Exception as e:
            logging.info(f'Error as occured : {e}')
            raise CustomException(e,sys)