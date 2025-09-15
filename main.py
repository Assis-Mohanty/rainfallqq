from terranova.components import data_transformation, model_trainer
from terranova.components.model_trainer import ModelTrainer
from terranova.components.data_ingestion import DataIngestion
from terranova.components.data_validation import DataValidation
from terranova.components.data_transformation import DataTransformation
from terranova.exceptions.exception import NetworkSecurityException
from terranova.logging.logger import logging
from terranova.entity.config_entity import DataIngestionConfig, DataTransformationConfig,DataValidationConfig, ModelTrainerConfig
from terranova.entity.artifact_entity import DataIngestionArtifact
from terranova.entity.config_entity import TrainingPipelineConfig
import sys

if __name__=='__main__':
    try:
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        logging.info("Initate the data ingestion")
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")
        print(data_ingestion_artifact)
        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact,data_validation_config)
        logging.info("Initated the data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_config)
        data_transformation_config=DataTransformationConfig(training_pipeline_config)
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("Initate the data transformation")
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")
        print(data_transformation_artifact)

        logging.info("Model Training started")
        model_trainer_config=ModelTrainerConfig(training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        logging.info("Model Training artifact created") 
    except Exception as e:
        raise NetworkSecurityException(e,sys)