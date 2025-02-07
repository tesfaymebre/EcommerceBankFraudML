import pandas as pd
import logging
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataMerger:
    def __init__(self, input_path: str, output_path: str):
        """
        Initializes the DataMerger class.

        Args:
            input_path (str): Path to the raw data directory.
            output_path (str): Path to save the merged dataset.
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def load_data(self, file_name: str):
        """
        Loads a dataset from a CSV file.

        Args:
            file_name (str): Name of the file to load.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        file_path = self.input_path / file_name
        logging.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)

    def merge_datasets(self, fraud_data: pd.DataFrame, ip_data: pd.DataFrame):
        """
        Merges Fraud_Data.csv with IpAddress_to_Country.csv based on IP address ranges.

        Args:
            fraud_data (pd.DataFrame): Fraud transaction dataset.
            ip_data (pd.DataFrame): IP address range dataset.

        Returns:
            pd.DataFrame: Merged dataset.
        """
        # Convert IP address columns to integers
        logging.info("Converting IP address columns to integers...")
        fraud_data["ip_address"] = fraud_data["ip_address"].astype(int)
        ip_data["lower_bound_ip_address"] = ip_data["lower_bound_ip_address"].astype(int)
        ip_data["upper_bound_ip_address"] = ip_data["upper_bound_ip_address"].astype(int)

        # Perform a range-based merge
        logging.info("Merging datasets based on IP address ranges...")
        merged_data = pd.merge_asof(
            fraud_data.sort_values("ip_address"),
            ip_data.sort_values("lower_bound_ip_address"),
            left_on="ip_address",
            right_on="lower_bound_ip_address",
            direction="backward"
        )

        logging.info(f"Merged dataset contains {merged_data.shape[0]} rows and {merged_data.shape[1]} columns.")
        return merged_data

    def save_data(self, data: pd.DataFrame, file_name: str):
        """
        Saves the merged dataset to a CSV file.

        Args:
            data (pd.DataFrame): Data to save.
            file_name (str): Name of the output file.
        """
        output_file_path = self.output_path / file_name
        logging.info(f"Saving merged dataset to {output_file_path}")
        data.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    # Example usage
    merger = DataMerger(input_path="data/raw", output_path="data/preprocessed")

    # Load datasets
    fraud_data = merger.load_data("Fraud_Data.csv")
    ip_data = merger.load_data("IpAddress_to_Country.csv")

    # Merge datasets
    merged_data = merger.merge_datasets(fraud_data, ip_data)

    # Save merged dataset
    merger.save_data(merged_data, "Fraud_Data_with_Geolocation.csv")
