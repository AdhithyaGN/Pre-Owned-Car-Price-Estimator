import sys
from dataclasses import dataclass


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import logging


import os
from src.utils import save_object


@dataclass
class DataTransformationConfig():
    preprocessor_obj_filepath:str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def makecoltransformer(self):
        try:
            num_list=['YOM', 'Selling Mileage', 'Buying Price', 'Total Actual RF', 'Warranty Charges',
                       'Insurance Charges','Age']
            
            cat_list=['Sale Type', 'Customer City', 'Model', 'Vehicle Sold Category', 'Finance/Cash']


            cat_pipeline=Pipeline(steps=[
                ('encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            num_pipeline=Pipeline(steps=[
                ('scaler',StandardScaler())


            ])


            logging.info(f"Categorical columns: {cat_list}")
            logging.info(f"Numerical columns: {num_list}")

            preprocessor=ColumnTransformer([
                ('catpipe',cat_pipeline,cat_list),
                ('numpipe',num_pipeline,num_list)])
            

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,data_path):

        try:


            data=pd.read_excel(data_path)
            

            logging.info("data loaded as dataframe")

            features=data.drop(columns=['POC Sales Date',
                                                'Date Of Registration','Vehicle Sell Price'],axis=1)
            labels=data['Vehicle Sell Price']
            
            

            

            logging.info('Applying preprocessing  data')

            prepocessing_obj=self.makecoltransformer()

            preprocessed_data=prepocessing_obj.fit_transform(features)
            

            data_arr=np.column_stack((preprocessed_data.toarray(),labels.to_numpy()))
            


            logging.info('Saving preprocessing object')

            save_object(file_path=self.data_transformation_config.preprocessor_obj_filepath,
                    obj=prepocessing_obj)
        

            return (data_arr,self.data_transformation_config.preprocessor_obj_filepath)
        except Exception as e:
            raise CustomException(e,sys)
    

    


        

















            


            




    
                   
                   








