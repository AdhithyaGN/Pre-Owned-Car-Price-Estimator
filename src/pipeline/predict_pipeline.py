import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_obj
import datetime
import os

class PredictPipeline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print('before loading')
            model=load_obj(file_path=model_path)
            preprocessor=load_obj(file_path=preprocessor_path)
            print("after loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,
        poc_sales_date:datetime.datetime,
                 
        Sale_Type:str,
        Customer_city:str,
        Model:str,
        Vehicle_sold_category:str,
        YOM:int,
        Date_of_Registration:datetime.datetime,
        Selling_Mileage:float,
        Buying_price:float,
        Total_Actual_RF:int,
        Warranty_charges:float,
        Insurance_charges:float,
        Finance_Cash:str
    ):
        self.poc_sales_data=poc_sales_date
        self.sale_type=Sale_Type
        self.Customer_city=Customer_city
        self.Model=Model
        self.Vehicle_sold_category=Vehicle_sold_category
        self.YOM=YOM
        self.Date_of_Registration=Date_of_Registration
        self.Selling_Mileage=Selling_Mileage
        self.Buying_Price=Buying_price
        self.Total_Actual_RF=Total_Actual_RF
        self.Warranty_Charges=Warranty_charges
        self.Insurance_Charges=Insurance_charges
        self.Finance_cash=Finance_Cash
    def get_data_as_data_frame(self):
        try:
            
            custom_data_input_dict={
                'POC Sales Date':[self.poc_sales_data],
                'Sale Type':[self.sale_type],
                'Customer City':[self.Customer_city],
                'Model':[self.Model],
                'Vehicle Sold Category':[self.Vehicle_sold_category],
                'YOM':[self.YOM],
                'Date Of Registration':[self.Date_of_Registration],
                'Selling Mileage':[self.Selling_Mileage],
                'Buying Price':[self.Buying_Price],
                'Total Actual RF':[self.Total_Actual_RF],
                'Warranty Charges':[self.Warranty_Charges],
                'Insurance Charges':[self.Insurance_Charges],
                'Finance/Cash':[self.Finance_cash]
                
            }
            data=pd.DataFrame(custom_data_input_dict)
            data['POC Sales Date']=pd.to_datetime(data['POC Sales Date'])
            data['Date Of Registration']=pd.to_datetime(data['Date Of Registration'])
            #data['POC Sales Date']=pd.to_datetime(data['POC Sales Date'],origin='1900-01-01',unit='D')
            #data['Date Of Registration']=pd.to_datetime(data['Date Of Registration'],origin='1900-01-01',unit='D')
            data['YOM'] = pd.to_numeric(data['YOM'], errors='coerce', downcast='integer')

            data['Age']=data['POC Sales Date']-data['Date Of Registration']
            data['Age'] = data['Age'].astype(str).str.extract('(\d+)').astype(int)
            data.drop(columns=['POC Sales Date',
                                                'Date Of Registration'],axis=1)
            
            return data



            
        
        except Exception as e:
            raise CustomException(e,sys)

                 
                 






























            


                 
                 
                 
                 






