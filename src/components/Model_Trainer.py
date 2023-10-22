import os 
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import eval_models,save_object
from sklearn.model_selection import train_test_split

@dataclass
class ModelTrainerConfig():
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,data_arr):
        try:

            train_set,test_set=train_test_split(data_arr,test_size=0.2,random_state=42)

            X_train,Y_train,X_test,Y_test=(train_set[:,:-1],train_set[:,-1],test_set[:,:-1],test_set[:,-1])


            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
                }
            
            logging.info("data splitted and models saves as a dictionary called models")
            
            report=eval_models(X_train,Y_train,X_test,Y_test,models=models)

            best_model_score=max(sorted(report.values()))

            best_model_name_loc=list(report.values()).index(best_model_score)

            best_model_name=list(report.keys())[best_model_name_loc]

            best_model=models[best_model_name]


            logging.info('best model got')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted=best_model.predict(X_test)

            r2=r2_score(Y_test,predicted)

            return r2


        except Exception as e:
            raise CustomException(e,sys)


        