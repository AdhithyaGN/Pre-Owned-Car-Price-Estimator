from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline,CustomData
import datetime


application=Flask(__name__)

app=application

### Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home2.html')
    else:
        data=CustomData(
            poc_sales_date=request.form.get("POCSaleDate"),
            Sale_Type=request.form.get("Saletype"),
            Customer_city=request.form.get("CustomerCity"),
            Model=request.form.get("Model"),
            Vehicle_sold_category=request.form.get("VehicleSoldCategory"),
            YOM=request.form.get("YOM"),
            Date_of_Registration=request.form.get("DateOfRegistration"),
            Selling_Mileage=request.form.get("SellingMileage"),
            Buying_price=request.form.get("BuyingPrice"),
            Total_Actual_RF=request.form.get("TotalActualRF"),
            Warranty_charges=request.form.get("WarrentyCharges"),
            Insurance_charges=request.form.get("InsuranceCharge"),
            Finance_Cash=request.form.get("Finance/Cash")
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before prdiction")


        predict_pipeline=PredictPipeline()
        print("mid prediction")
        results=predict_pipeline.predict(pred_df)
        print('after prediction')

        return render_template('home.html',results=f"Rs {round(results[0])}")
    
if __name__=="__main__":
        app.run(host="0.0.0.0")
        














        


