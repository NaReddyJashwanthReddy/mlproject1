import pandas as pd 
import numpy as np 
import sys
import os

from src.exception import CustomException
from src.utils import load_obj

from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass
    
    def Predict_score(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_obj(file_path=model_path)
            preprocessor=load_obj(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            predict=model.predict(data_scaled)
            return predict
        except Exception as e:
            raise CustomException(e,sys)
        
    
class CustomData:  # responsible for maping all the html inputs
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score: int,
                 writing_score: int
                 ):
        
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
        
    def get_data_as_frame(self):
        logging.info("started the preparation of input data")
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)