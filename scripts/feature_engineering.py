import pandas as pd
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FeatureEngineer:
    """
    A class to handle feature engineering tasks such as time-based feature creation and derived features.
    """

    @staticmethod
    def create_time_features(data, time_column):
        """
        Creates time-based features like 'hour_of_day' and 'day_of_week' from a datetime column.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing the datetime column.
        time_column : str
            The name of the datetime column.

        Returns:
        --------
        pd.DataFrame
            Dataset with new time-based features.
        """
        try:
            data['hour_of_day'] = data[time_column].dt.hour
            data['day_of_week'] = data[time_column].dt.dayofweek
            logging.info(f"Time-based features created: 'hour_of_day', 'day_of_week' from {time_column}.")
            return data
        except Exception as e:
            logging.error(f"Error creating time-based features from {time_column}: {e}")
            return data

    @staticmethod
    def create_transaction_features(data, user_id_column, transaction_time_column):
        """
        Creates transaction frequency and velocity features for each user.
        """
        try:
            # Ensure datetime format
            data[transaction_time_column] = pd.to_datetime(data[transaction_time_column])

            # Transaction count per user
            data['transaction_count'] = data.groupby(user_id_column)[transaction_time_column].transform('count')

            # Time span between first and last transaction per user (in days)
            data['days_since_first_transaction'] = (
                data.groupby(user_id_column)[transaction_time_column]
                .transform(lambda x: (x.max() - x.min()).days + 1)
            )

            # Transaction velocity: transactions per day per user
            data['transaction_velocity'] = data['transaction_count'] / data['days_since_first_transaction']

            logging.info("Transaction frequency and velocity features created.")
            return data

        except Exception as e:
            logging.error(f"Error creating transaction features: {e}")
            return data
