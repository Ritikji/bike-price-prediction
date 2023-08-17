import sys
import os
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    
    def predict(self,features):
        try:
            logging.info("Predict started")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            data_scaled = preprocessor.transform(features)
            
            pred=model.predict(data_scaled)
            return pred
            
            
            
            
            
            
        except Exception as e:
            logging.info("Error in predict from prediction_pipeline")
            raise CustomException(e,sys)
            

class CustomData:
    def __init__(self,
            bike_name:str,
            city:str,
            kms_driven:float,
            owner:int,
            age:float,
            power:float,
            brand:int):

        self.bike_name = bike_name
        self.city = city
        self.kms_driven = kms_driven
        self.owner = owner
        self.age  = age
        self.power = power
        self.brand = brand
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "bike_name":[self.bike_name],
                "city":[self.city],
                "kms_driven":[self.kms_driven],
                "owner":[self.owner],
                "age":[self.age],
                "power":[self.power],
                "brand":[self.brand],
            }

            data = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return data

        except Exception as e:
            logging.info("Error Occured In Prediction Pipline")
            raise CustomException(e, sys)
   



  