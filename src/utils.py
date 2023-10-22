import os
import sys

import numpy as np
import dill
import pickle
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        directory=os.path.dirname(file_path)
        os.makedirs(directory,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def eval_models(X_train,Y_train,X_test,Y_test,models:dict):
    try:
        report={}


        for i in range(len(models)):
            model=list(models.values())[i]

            model.fit(X_train,Y_train)

            Y_train_pred=model.predict(X_train)

            Y_test_pred=model.predict(X_test)

            train_model_score=r2_score(Y_train,Y_train_pred)

            test_model_score=r2_score(Y_test,Y_test_pred)

            report[list(models.keys())[i]]=test_model_score



        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    

def load_obj(file_path):

    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)





