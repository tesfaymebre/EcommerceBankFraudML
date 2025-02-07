import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataCleaner:
    def __init__(self):
        """
        Initializes the DataCleaner class.
        """
        self.scaler = MinMaxScaler()

    def handle_outliers(self, data: pd.DataFrame, column: str, method="cap", threshold=1.5):
        """
        Handles outliers in a column using capping, flooring, or log transformation.

        Parameters:
        - data (pd.DataFrame): Dataset.
        - column (str): Column name.
        - method (str): Method to handle outliers ('cap', 'log').
        - threshold (float): Threshold for IQR method (used for capping).
        """
        if method == "cap":
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            data[column] = data[column].clip(lower_bound, upper_bound)
            logging.info(f"Capped outliers in {column} to range [{lower_bound}, {upper_bound}].")
        elif method == "log":
            data[column] = data[column].apply(lambda x: np.log1p(x) if x > 0 else x)
            logging.info(f"Applied log transformation to {column}.")
        return data

    def normalize_columns(self, data: pd.DataFrame, columns: list):
        """
        Normalizes specified columns using Min-Max scaling.

        Parameters:
        - data (pd.DataFrame): Dataset.
        - columns (list): List of column names to normalize.
        """
        data[columns] = self.scaler.fit_transform(data[columns])
        logging.info(f"Normalized columns: {columns}.")
        return data

    def encode_categorical(self, data: pd.DataFrame, drop_first=True):
        """
        Encodes categorical columns using one-hot encoding.

        Parameters:
        - data (pd.DataFrame): Dataset.
        - drop_first (bool): Whether to drop the first category to avoid multicollinearity.
        """
        categorical_cols = data.select_dtypes(include="object").columns
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=drop_first)
        logging.info(f"Encoded categorical columns: {categorical_cols}.")
        return data

    def create_features(self, data: pd.DataFrame):
        """
        Creates additional features (e.g., time_since_signup, temporal features).

        Parameters:
        - data (pd.DataFrame): Dataset.
        """
        if "signup_time" in data.columns and "purchase_time" in data.columns:
            data["signup_time"] = pd.to_datetime(data["signup_time"])
            data["purchase_time"] = pd.to_datetime(data["purchase_time"])
            data["time_since_signup"] = (data["purchase_time"] - data["signup_time"]).dt.total_seconds() / 3600
            logging.info("Created 'time_since_signup' feature.")

        if "Time" in data.columns:
            data["hour_of_day"] = (data["Time"] % (3600 * 24)) // 3600
            logging.info("Created 'hour_of_day' feature from 'Time'.")
        return data

    def clean_data(self, data: pd.DataFrame, outlier_columns: list, normalize_columns: list):
        """
        Performs full data cleaning steps: outlier handling, normalization, and encoding.

        Parameters:
        - data (pd.DataFrame): Dataset.
        - outlier_columns (list): List of columns to handle outliers.
        - normalize_columns (list): List of columns to normalize.

        Returns:
        - pd.DataFrame: Cleaned dataset.
        """
        for col in outlier_columns:
            data = self.handle_outliers(data, column=col, method="cap")
        
        data = self.normalize_columns(data, columns=normalize_columns)
        data = self.encode_categorical(data)
        return data
