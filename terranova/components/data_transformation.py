import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline 

from terranova.constants.training_pipeline import TARGET_COLUMN
from terranova.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from terranova.entity.artifact_entity import(
    DataTransformationArtifact,DataValidationArtifact
)
from terranova.exceptions.exception import NetworkSecurityException
from terranova.entity.config_entity import DataTransformationConfig 
from terranova.logging.logger import logging
from terranova.utils.main_utils.utils import save_numpy_array,save_object



class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config

        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def get_transformation_object(input_df) -> Pipeline:
        """
        Creates a preprocessing pipeline for the phishing dataset (all numeric columns).
        Only uses columns present in both the schema and the DataFrame.
        """
        logging.info("Entered get_transformation_object method of Transformation class")
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import KNNImputer
            import yaml
            with open('data_schema/schema.yaml', 'r') as f:
                schema = yaml.safe_load(f)
            schema_numeric_cols = schema.get('numerical_columns', [])
            # Remove target column if present
            for col in ['Yield', 'yield', 'Result', 'target']:
                if col in schema_numeric_cols:
                    schema_numeric_cols.remove(col)
            df_cols = input_df.columns.tolist()
            print(f"[DEBUG] DataFrame columns: {df_cols}")
            print(f"[DEBUG] Schema numeric columns: {schema_numeric_cols}")
            # Only use columns present in both
            numeric_cols = [col for col in schema_numeric_cols if col in df_cols]
            print(f"[DEBUG] Numeric columns used for pipeline (intersection): {numeric_cols}")
            numeric_pipeline = Pipeline(steps=[
                ('imputer', KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ('scaler', StandardScaler())
            ])
            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_pipeline, numeric_cols)
            ])
            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered the initiate_data_transformation of the DataTransformation class")
        try:
            logging.info("Starting the data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            #training dataframe
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df=train_df[TARGET_COLUMN]
            target_feature_train_df=target_feature_train_df.replace(-1,0)

            #test dataframe
            input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df=test_df[TARGET_COLUMN]
            target_feature_test_df=target_feature_test_df.replace(-1,0)
            


            # Use the static method correctly
            preprocessor = DataTransformation.get_transformation_object(input_feature_train_df)
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)


            # Debug: print shapes before concatenation
            print('Shape of transformed_input_train_feature:', transformed_input_train_feature.shape)
            print('Shape of target_feature_train_df:', np.array(target_feature_train_df).shape)
            print('Shape of transformed_input_test_feature:', transformed_input_test_feature.shape)
            print('Shape of target_feature_test_df:', np.array(target_feature_test_df).shape)

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object)
            save_numpy_array(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path,array=test_arr)
            
            save_object("final_model/preprocessing.pkl",preprocessor_object)
            

            #preparing artifacts
            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path 
            )
            return data_transformation_artifact 
        except Exception as e:
            raise NetworkSecurityException(e, sys)