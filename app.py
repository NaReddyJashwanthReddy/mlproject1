import numpy as np 
import pandas as pd 
from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging

from sklearn.preprocessing import StandardScaler  
from flask import Flask,request,render_template
application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        

        pred_df=data.get_data_as_frame()
        print(pred_df)
        
        predict_pipeline=PredictPipeline()
        prediction=predict_pipeline.Predict_score(pred_df)
        logging.info(f"predicted the math score{prediction}")
        return render_template('home.html',results=int(prediction[0]))
        
        
if __name__=='__main__':
    app.run(host="0.0.0.0")
        