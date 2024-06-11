import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging  # Assuming you have a logging module

# Ensure the project root directory is added to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)

class PredictPipeline:
    def __init__(self):
        logging.info("PredictPipeline initialization")
        
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")
            
            logging.info("Before loading model and preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Model and preprocessor loaded successfully")
            
            logging.info("Before data transformation")
            data_scaled = preprocessor.transform(features)
            logging.info("Data transformation completed")
            
            logging.info("Before model prediction")
            preds = model.predict(data_scaled)
            logging.info("Model prediction completed")
            
            return preds
        
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {e}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education, lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            logging.info("Custom data converted to DataFrame")
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            logging.error(f"Error in CustomData.get_data_as_data_frame: {e}")
            raise CustomException(e, sys)
