import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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


    @staticmethod
    def create_high_risk_country_flag(data, country_column, high_risk_countries):
        """
        Creates a binary flag for transactions from high-risk fraud countries.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing country information.
        country_column : str
            The column representing the country.

        Returns:
        --------
        pd.DataFrame
            Dataset with a new column 'high_risk_country' (1 = high risk, 0 = low risk).
        """
        try:

            # Create a binary flag for high-risk countries
            data['high_risk_country'] = data[country_column].apply(lambda x: 1 if x in high_risk_countries else 0)

            logging.info(f"Created 'high_risk_country' feature using {len(high_risk_countries)} high-fraud countries.")
            return data

        except Exception as e:
            logging.error(f"Error creating high-risk country flag: {e}")
            return data

    @staticmethod
    def normalize_numerical_features(data, columns):
        """
        Normalizes numerical features using Min-Max Scaling.
        """
        try:
            scaler = MinMaxScaler()
            data[columns] = scaler.fit_transform(data[columns])
            logging.info(f"Normalized columns: {columns}.")
            return data
        except Exception as e:
            logging.error(f"Error normalizing numerical features: {e}")
            return data

    @staticmethod
    def encode_categorical_features(data, categorical_columns):
        """
        Encodes categorical features using One-Hot Encoding.
        """
        try:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_df = pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
            encoded_df.columns = encoder.get_feature_names_out(categorical_columns)

            # Drop original categorical columns and add encoded ones
            data = data.drop(columns=categorical_columns).reset_index(drop=True)
            data = pd.concat([data, encoded_df], axis=1)

            logging.info(f"Encoded categorical features: {categorical_columns}.")
            return data
        except Exception as e:
            logging.error(f"Error encoding categorical features: {e}")
            return data
