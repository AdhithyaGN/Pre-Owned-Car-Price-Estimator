import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import re
import numpy as np
from src.components.Data_Transformation import DataTransformationConfig
from src.components.Data_Transformation import DataTransformation
from src.components.Model_Trainer import ModelTrainer,ModelTrainerConfig

@dataclass()
class  DataIngestionConfig:
    data_path:str=os.path.join('artifacts','data.xlsx')
    


class DataIngestion:
    def __init__(self):
        self.ingestionconfig=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')

        try:
            data=pd.read_excel('notebook\True_Value_data.xlsx')
            logging.info('Read the data as Dataframe')


            data.dropna(thresh=10,inplace=True)
            logging.info('Row with many NAN Values removed sucessfully')

            data['Date Of Registration']=pd.to_datetime(data['Date Of Registration'],origin='1900-01-01',unit='D')

            logging.info('Date of Registration column datatype changed')

            data['YOM'] = pd.to_numeric(data['YOM'], errors='coerce', downcast='integer')
            logging.info('YOM column data type changed')


            data['Age']=data['POC Sales Date']-data['Date Of Registration']
            data['Age'] = data['Age'].astype(str).str.extract('(\d+)').astype(int)
            logging.info('Age column added and transformed')

            pattern = r"\[[^\]]*\]|\([^)]*\)"
            for i in range(len(data['Customer City'])):
                review=re.sub(pattern,"",data['Customer City'][i])
                data['Customer City'][i]=review
            logging.info("unwanted characters removed from Customer city column")

            data['Model'].replace('-',np.nan,inplace=True)
            data['Model'].fillna(method='bfill',inplace=True)
            logging.info('hyphen values removed from  Model column')

            data['Total Actual RF'] = pd.to_numeric(data['Total Actual RF'], errors='coerce', downcast='integer')
            data['Total Actual RF']=data['Total Actual RF'].fillna(data['Total Actual RF'].mean())
            logging.info('Total Actual RF column changed its datatype to float and nan values cleared')

            data['Finance/Cash'].replace('-',data['Finance/Cash'].mode()[0],inplace=True)
            logging.info('hyphen values removed from  Finance/cash column')

        
            os.makedirs(os.path.dirname(self.ingestionconfig.data_path),exist_ok=True)

            data.to_excel(self.ingestionconfig.data_path,index=False,header=True)

            

            

            

            

            logging.info('Data Ingestion successfully done')

            return(
                self.ingestionconfig.data_path
                
            )





        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    data_path=obj.initiate_data_ingestion()

    transformation_obj=DataTransformation()
    data_arr,_=transformation_obj.initiate_data_transformation(data_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(data_arr=data_arr))









