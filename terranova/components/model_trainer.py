import os
import sys

from terranova.exceptions.exception import NetworkSecurityException
from terranova.logging.logger import logging

from terranova.entity.artifact_entity import DataTransformationArtifact,ClassificationMatricArtifact,ModelTrainerArtifact
from terranova.entity.config_entity import ModelTrainerConfig

from terranova.utils.main_utils.utils import evaluate_models, save_object,load_object
        
from terranova.utils.main_utils.utils import load_numpy_array_data
from terranova.utils.ml_utils import model
from terranova.utils.ml_utils.metric.classification_metric import get_classification_metrics
from terranova.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
import mlflow

import dagshub
dagshub.init(repo_owner='Assis-Mohanty', repo_name='crossOrgin', mlflow=True)


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self,best_model,classification_metric):
        with mlflow.start_run():
            f1_score=classification_metric.f1_score
            precision_score=classification_metric.precision_score
            recall_score=classification_metric.recall_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")




    def train_model(self,x_train,y_train,x_test,y_test):
        models={
            "Random Forest":RandomForestRegressor(),
            "Decision Tree":DecisionTreeRegressor(),
            "Gradient Boosting":GradientBoostingRegressor(),
            "Linear Regression":LinearRegression(),
            "AdaBoost":AdaBoostRegressor(),
        }

        params={
            "Decision Tree":{
                'max_depth':[None, 5, 10, 20],
                'min_samples_split':[2, 5, 10]
            },
            "Random Forest":{
                'n_estimators':[8,16,32,64,128,256],
                'max_depth':[None, 5, 10, 20]
            },
            "Gradient Boosting":{
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                'n_estimators' :[8,16,32,64,128,256]
            },
            "Linear Regression":{},
            "AdaBoost":{
                'learning_rate':[0.1,0.01,0.5,0.001],
                'n_estimators' :[8,16,32,64,128,256]
            }
        }
        model_report:dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, params=params)
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]
        best_model.fit(x_train, y_train)
        y_train_pred = best_model.predict(x_train)
        y_test_pred = best_model.predict(x_test)

        # Regression metrics
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        # Log metrics to MLflow
        import pickle
        with mlflow.start_run():
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_mse", test_mse)
            # Save model to file and log as artifact (DagsHub MLflow workaround)
            with open("model.pkl", "wb") as f:
                pickle.dump(best_model, f)
            mlflow.log_artifact("model.pkl")
            os.remove("model.pkl")

        preprocssor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        Network_Model = NetworkModel(preprocessor=preprocssor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)
        save_object("final_model/model.pkl", best_model)

        # You may want to create a regression metric artifact, but for now, return the scores
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact={"r2": train_r2, "mae": train_mae, "mse": train_mse},
            test_metric_artifact={"r2": test_r2, "mae": test_mae, "mse": test_mse}
        )
        logging.info(f"Model trainer artifact :{model_trainer_artifact}")
        return model_trainer_artifact



    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
              train_file_path=self.data_transformation_artifact.transformed_train_file_path
              test_file_path=self.data_transformation_artifact.transformed_test_file_path
              
              train_arr=load_numpy_array_data(train_file_path)
              test_arr=load_numpy_array_data(test_file_path)

              x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
              )
              model_trainer_artifact=self.train_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
              return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
