from pathlib import Path
import os
import yaml

from crop_yield_prediction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataPreprocessingConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)


class ConfigManager:

    def __init__(self, config_path: Path):
        self.config = self._read_yaml(config_path)


    def _read_yaml(self, config_path: Path):
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r") as file:
            return yaml.safe_load(file)



    def get_data_ingestion_config(self) -> DataIngestionConfig:

        ingestion = self.config.get("data_ingestion")
        if ingestion is None:
            raise ValueError("data_ingestion section missing in config.yaml")

        root_dir = Path(ingestion["root_dir"])
        os.makedirs(root_dir, exist_ok=True)

        return DataIngestionConfig(
            root_dir=root_dir,
            source_dir=Path(ingestion["source_dir"]),
            train_dir=Path(ingestion["train_dir"]),
            test_dir=Path(ingestion["test_dir"]),
        )



    def get_data_validation_config(self) -> DataValidationConfig:

        validation = self.config.get("data_validation")
        if validation is None:
            raise ValueError("data_validation section missing in config.yaml")

        root_dir = Path(validation["root_dir"])
        os.makedirs(root_dir, exist_ok=True)

        return DataValidationConfig(
            root_dir=root_dir,
            validation_status_file=Path(validation["validation_status_file"]),
            train_dir=Path(validation["train_dir"]),
            schema_file=Path(validation["schema_file"]),
        )



    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:

        preprocessing = self.config.get("data_preprocessing")
        ingestion = self.config.get("data_ingestion")

        if preprocessing is None:
            raise ValueError("data_preprocessing section missing in config.yaml")

        root_dir = Path(preprocessing["root_dir"])
        os.makedirs(root_dir, exist_ok=True)

        return DataPreprocessingConfig(
            root_dir=root_dir,
            train_dir=Path(ingestion["train_dir"]),
            test_dir=Path(ingestion["test_dir"]),
            processed_train_dir=Path(preprocessing["processed_train_dir"]),
            processed_test_dir=Path(preprocessing["processed_test_dir"]),
            scaler_path=Path(preprocessing["scaler_path"]),
        )



    def get_model_trainer_config(self) -> ModelTrainerConfig:

        trainer = self.config.get("model_trainer")
        preprocessing = self.config.get("data_preprocessing")

        if trainer is None:
            raise ValueError("model_trainer section missing in config.yaml")

        root_dir = Path(trainer["root_dir"])
        os.makedirs(root_dir, exist_ok=True)

        return ModelTrainerConfig(
            root_dir=root_dir,
            processed_train_dir=Path(preprocessing["processed_train_dir"]),
            processed_test_dir=Path(preprocessing["processed_test_dir"]),
            model_path=Path(trainer["model_path"]),
            params_file=Path(trainer["params_file"]),
        )



    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        evaluation = self.config.get("model_evaluation")

        if evaluation is None:
            raise ValueError("model_evaluation section missing in config.yaml")

        root_dir = Path(evaluation["root_dir"])
        os.makedirs(root_dir, exist_ok=True)

        return ModelEvaluationConfig(
            root_dir=root_dir,
            model_path=Path(evaluation["model_path"]),
            processed_test_dir=Path(evaluation["processed_test_dir"]),
            metrics_file=Path(evaluation["metrics_file"]),
        )