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
        

    def initiate_data_transformation(self,train_path,test_path):

        try:


            train_data=pd.read_excel(train_path)
            test_data=pd.read_excel(test_path)

            logging.info("train and test data loaded as dataframe")

            train_features=train_data.drop(columns=['POC Sales Date',
                                                'Date Of Registration','Vehicle Sell Price'],axis=1)
            test_features=test_data.drop(columns=['POC Sales Date',
                                                'Date Of Registration','Vehicle Sell Price'],axis=1)
            train_labels=train_data['Vehicle Sell Price']

            test_labels=test_data['Vehicle Sell Price']

            logging.info('Applying preprocessing obj on training and test data')

            prepocessing_obj=self.makecoltransformer()

            preprocessed_train_data=prepocessing_obj.fit_transform(train_features)
            preprocessed_test_data=prepocessing_obj.fit_transform(test_features)

            train_arr=np.column_stack((preprocessed_train_data.toarray(),train_labels.to_numpy()))
            test_arr=np.column_stack((preprocessed_test_data.toarray(),test_labels.to_numpy()))


            logging.info('Saving preprocessing object')

            save_object(file_path=self.data_transformation_config.preprocessor_obj_filepath,
                    obj=prepocessing_obj)
        

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_filepath)
        except Exception as e:
            raise CustomException(e,sys)
    

    


        

















            


            




    
                   
                   








