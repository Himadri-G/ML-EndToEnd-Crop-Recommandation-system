import os
from pathlib import Path
import joblib

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from crop_yield_prediction.entity.config_entity import DataPreprocessingConfig
from crop_yield_prediction.utils.logger import get_logger

logger = get_logger(
    name=__name__,
    log_file="data_preprocessing.log"
)

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.preprocessor = None  # Will hold the ColumnTransformer

    def _load_data(self):
        logger.info("Loading train and test data")
        train_df = pd.read_csv(self.config.train_dir)
        test_df = pd.read_csv(self.config.test_dir)

        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Test shape: {test_df.shape}")

        return train_df, test_df

    def _split_features_target(self, df):
        X = df.drop(columns=["hg/ha_yield"])
        y = df["hg/ha_yield"]
        return X, y

    def main_data_preprocessing(self):
        logger.info("Starting Data Preprocessing!")

        # Load train and test
        train_df, test_df = self._load_data()
        X_train, y_train = self._split_features_target(train_df)
        X_test, y_test = self._split_features_target(test_df)

        # Identify numeric and categorical columns automatically
        numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

        logger.info(f"Numeric columns: {numeric_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")

        # Define ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols)
            ]
        )

        # Fit-transform train, transform test
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Get proper column names
        ohe_cols = []
        if categorical_cols:
            ohe_cols = self.preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
        all_cols = numeric_cols + list(ohe_cols)

        # Create DataFrames
        train_processed = pd.DataFrame(X_train_processed, columns=all_cols)
        train_processed["hg/ha_yield"] = y_train.values

        test_processed = pd.DataFrame(X_test_processed, columns=all_cols)
        test_processed["hg/ha_yield"] = y_test.values

        # Ensure directory exists
        Path(self.config.preprocessed_train_dir).parent.mkdir(parents=True, exist_ok=True)

        # Save processed data
        train_processed.to_csv(self.config.preprocessed_train_dir, index=False)
        test_processed.to_csv(self.config.preprocessed_test_dir, index=False)

        # Save preprocessor
        joblib.dump(self.preprocessor, self.config.scaler_path)

        logger.info("Data Preprocessing Completed!")

        return self.config.preprocessed_train_dir, self.config.preprocessed_test_dir