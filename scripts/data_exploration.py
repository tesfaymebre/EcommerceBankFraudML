import numpy as np
import logging
import os
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EDA:
    def __init__(self, input_path):
        """
        Initialize the EDA class with input path.
        """
        self.input_path = input_path

    def load_data(self, file_name):
        """
        Loads a CSV file into a Pandas DataFrame.
        """
        try:
            file_path = os.path.join(self.input_path, file_name)
            data = pd.read_csv(file_path)
            logging.info(f"Successfully loaded {file_name}")
            return data
        except Exception as e:
            logging.error(f"Error loading {file_name}: {e}")
            return None

    def get_summary(self, data, file_name):
        """
        Provides a summary of the dataset: missing values, data types, and basic statistics.
        """
        try:
            logging.info(f"Generating summary for {file_name}")
            summary = {
                "Shape": data.shape,
                "Missing Values": data.isnull().sum(),
                "Data Types": data.dtypes,
                "Statistics": data.describe(include="all")
            }
            return summary
        except Exception as e:
            logging.error(f"Error generating summary for {file_name}: {e}")
            return None

    def check_duplicates(self, data, file_name):
        """
        Checks for duplicate rows in the dataset.
        """
        try:
            duplicates = data.duplicated().sum()
            logging.info(f"{file_name} contains {duplicates} duplicate rows.")
            return duplicates
        except Exception as e:
            logging.error(f"Error checking duplicates for {file_name}: {e}")
            return None

    def detect_outliers(self, data, column, method="iqr", z_threshold=3, cap=False):
        """
        Detects outliers in a given column using Z-score or IQR methods.
        
        Parameters:
        - data: Pandas DataFrame.
        - column: Column to check for outliers.
        - method: "iqr" for Interquartile Range or "zscore" for Z-score.
        - z_threshold: Threshold for Z-score outlier detection.

        Returns:
        - outlier_indices: Indices of detected outliers.
        """
        try:
            if method == "iqr":
                q1 = data[column].quantile(0.25)
                q3 = data[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)].index

            elif method == "zscore":
                mean = data[column].mean()
                std = data[column].std()
                z_scores = (data[column] - mean) / std
                outliers = data[np.abs(z_scores) > z_threshold].index

            if cap:
                if method == "iqr":
                    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
                    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
                elif method == "zscore":
                    data[column] = np.where(z_scores < -z_threshold, mean - z_threshold * std, data[column])
                    data[column] = np.where(z_scores > z_threshold, mean + z_threshold * std, data[column])
                logging.info(f"Capped outliers in {column} to range [{lower_bound}, {upper_bound}]")
            else:
                logging.info(f"Outliers detected in {column} using {method} method: {len(outliers)}")
            return outliers
        except Exception as e:
            logging.error(f"Error detecting outliers in {column}: {e}")
            return None
        outliers = eda.detect_outliers(fraud_data, column="purchase_value", method="iqr", cap=True)

if __name__ == "__main__":
    # Example usage
    eda = EDA(input_path="data/raw")
    fraud_data = eda.load_data("Fraud_Data.csv")
    if fraud_data is not None:
        print(eda.get_summary(fraud_data, "Fraud_Data.csv"))
        print(f"Duplicate Rows: {eda.check_duplicates(fraud_data, 'Fraud_Data.csv')}")
        outliers = eda.detect_outliers(fraud_data, column="purchase_value", method="iqr")
        print(f"Outliers in 'purchase_value': {len(outliers)} rows")
