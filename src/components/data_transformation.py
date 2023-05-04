import sys
import os
from dataclasses import dataclass 
from src.utils import save_obj

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformer_config=DataTransformationConfig()  
    
    def get_data_transformation_obj(self):
        '''
        This function is used for preprocessing the data
        '''
        logging.info("initialise the data preprocessng")
        try:
            numerical_feature=[ 
                'reading_score',
                'writing_score'
                ]
            categorical_feature=[
                'gender', 
                'race_ethnicity',
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
                ]
            
            logging.info(f"numerical columns : {numerical_feature}")
            logging.info(f"categorical feaures : {categorical_feature}")
            
            num_pipeline=Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            )
            
            logging.info("numerical columns has scaled")
            
            cat_pipeline=Pipeline(
                steps=[
                    ("impte",SimpleImputer(strategy='most_frequent')),
                    ("ohe",OneHotEncoder(drop='first'))
                ]
            )
            
            logging.info(" Categorical columns encoding completed")
            
            
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_feature),
                    ('cat_pipeline',cat_pipeline,categorical_feature)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        logging.info("initialise the data transforming")
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Reading the train and test data complete")
            
            logging.info("obtaining preprocessing objects")
            
            preprocessing_obj=self.get_data_transformation_obj()

            
            target_column=['math_score']
            numeric_columns=['writing_score','reading_score']
            
            input_feature_train_df=train_df.drop(columns=target_column,axis=1)
            input_feature_test_df=test_df.drop(columns=target_column,axis=1)
            
            target_feature_train_df=train_df['math_score']
            target_feature_test_df=test_df['math_score']
            
            logging.info("seperated the dependent and independent features")
            logging.info(f"input features : {input_feature_train_df.columns}")
            logging.info(f"output features : {target_feature_train_df.shape} ")
            
            preprocessing_obj_tarin_df=preprocessing_obj.fit_transform(input_feature_train_df)
            preprocessing_obj_test_df=preprocessing_obj.transform(input_feature_test_df)
            
            
            logging.info("preprocessing the inputs are done")
            
            train_arr=np.c_[
                preprocessing_obj_tarin_df,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                preprocessing_obj_test_df,np.array(target_feature_test_df)
            ]
            
            logging.info("saved preprocessing object.")
            
            
            save_obj(
                file_path=self.data_transformer_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
