import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataCleaner:
    def __init__(self, input_path=None, output_path=None):
        """
        Initializes the DataCleaner class.

        Args:
            input_path (str): Path to the input data directory.
            output_path (str): Path to save the cleaned data.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.scaler = MinMaxScaler()

    def load_data(self, file_name):
        """
        Loads a dataset from the input directory.

        Args:
            file_name (str): Name of the file to load.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        file_path = f"{self.input_path}/{file_name}"
        logging.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data with shape {data.shape}")
        return data

    def save_data(self, data, file_name):
        """
        Saves the dataset to the output directory.

        Args:
            data (pd.DataFrame): Dataset to save.
            file_name (str): Name of the output file.
        """
        file_path = f"{self.output_path}/{file_name}"
        logging.info(f"Saving cleaned data to {file_path}")
        data.to_csv(file_path, index=False)

    def convert_data_types(self, data, columns, dtype="datetime"):
        """
        Converts the specified columns to the desired data type.

        Args:
            data (pd.DataFrame): Dataset.
            columns (list): List of column names to convert.
            dtype (str): Data type to convert to ("datetime", "int", "float", etc.).

        Returns:
            pd.DataFrame: Dataset with updated data types.
        """
        for col in columns:
            if dtype == "datetime":
                data[col] = pd.to_datetime(data[col], errors="coerce")
                logging.info(f"Converted column {col} to datetime format.")
            elif dtype == "int":
                data[col] = data[col].astype(int, errors="ignore")
                logging.info(f"Converted column {col} to integer format.")
            elif dtype == "float":
                data[col] = data[col].astype(float, errors="ignore")
                logging.info(f"Converted column {col} to float format.")
        return data
    
    def remove_duplicates(self, data):
        """
        Removes duplicate rows from the dataset.

        Args:
            data (pd.DataFrame): Dataset.

        Returns:
            pd.DataFrame: Dataset with duplicate rows removed.
        """
        initial_shape = data.shape
        data.drop_duplicates(inplace=True)
        logging.info(f"Removed {initial_shape[0] - data.shape[0]} duplicate rows.")
        return data

    def handle_outliers(self, df, column_method, threshold=1.5):
        """
        Handles outliers in a column using capping, flooring, or log transformation.

        Args:
            data (pd.DataFrame): Dataset.
            column_methos list(str,str): Column name and method to handle outliers ("iqr","log").
            threshold (float): Threshold for IQR method (used for capping).
        """

        for col,method in column_method:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                logging.info(f"Removed outliers in {col} using IQR method.")
            elif method == "zscore":
                from scipy.stats import zscore
                df = df[(zscore(df[col]) < 3).all(axis=1)]
                logging.info(f"Removed outliers in {col} using Z-score method.")
            elif method == "log":
                df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
                logging.info(f"Applied log transformation to {col}.")

        return df

    def normalize_columns(self, data, columns):
        """
        Normalizes specified columns using Min-Max scaling.

        Args:
            data (pd.DataFrame): Dataset.
            columns (list): List of column names to normalize.
        """
        data[columns] = self.scaler.fit_transform(data[columns])
        logging.info(f"Normalized columns: {columns}.")
        return data

    def encode_categorical(self, data, drop_first=True, cardinality_threshold=50):
        """
        Encodes categorical columns using one-hot encoding.

        Args:
            data (pd.DataFrame): Dataset.
            drop_first (bool): Whether to drop the first category to avoid multicollinearity.
            cardinality_threshold (int): Maximum unique values allowed for one-hot encoding.
        """
        categorical_cols = [
            col for col in data.select_dtypes(include="object").columns
            if not pd.api.types.is_datetime64_any_dtype(data[col])
        ]

        for col in categorical_cols:
            unique_values = data[col].nunique()
            if unique_values <= cardinality_threshold:
                logging.info(f"Encoding column {col} with {unique_values} unique values.")
                data = pd.get_dummies(data, columns=[col], drop_first=drop_first)
            else:
                logging.warning(f"Skipping column {col} with {unique_values} unique values (high cardinality).")
        return data

    def create_features(self, data):
        """
        Creates additional features (e.g., time_since_signup, temporal features).

        Args:
            data (pd.DataFrame): Dataset.
        """
        if "signup_time" in data.columns and "purchase_time" in data.columns:
            data["time_since_signup"] = (
                pd.to_datetime(data["purchase_time"]) - pd.to_datetime(data["signup_time"])
            ).dt.total_seconds() / 3600
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
