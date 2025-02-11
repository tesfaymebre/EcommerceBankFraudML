import pandas as pd
import socket
import struct
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GeolocationAnalyzer:
    """
    A class to handle geolocation-based analysis and merging of IP address data.
    """

    @staticmethod
    def ip_to_int(ip):
        """
        Convert an IP address to its integer representation.

        Parameters:
        ----------
        ip : str
            IP address in string format (e.g., "192.168.1.1").

        Returns:
        --------
        int or None
            Converted integer representation of the IP address or None if invalid.
        """
        try:
            return struct.unpack("!I", socket.inet_aton(ip))[0]
        except socket.error:
            return None  # Handle invalid IPs gracefully

    @staticmethod
    def merge_fraud_with_geolocation(fraud_data, ip_data):
        """
        Merges fraud data with geolocation data based on IP address ranges.

        Parameters:
        ----------
        fraud_data : pd.DataFrame
            The fraud dataset containing an 'ip_address' column.
        ip_data : pd.DataFrame
            The geolocation dataset containing 'lower_bound_ip_address' and 'upper_bound_ip_address'.

        Returns:
        --------
        pd.DataFrame
            Merged dataset including country information.
        """
        try:
            logging.info("Converting IP addresses to integers...")

            # Convert fraud IPs to integer format
            fraud_data['ip_int'] = fraud_data['ip_address'].apply(lambda x: GeolocationAnalyzer.ip_to_int(str(int(x))) if not pd.isna(x) else None)

            # Drop invalid IPs
            fraud_data.dropna(subset=['ip_int'], inplace=True)

            # Convert lower and upper bound IPs to integer
            ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(int)
            ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(int)

            # Sort both datasets for merge_asof
            fraud_data.sort_values('ip_int', inplace=True)
            ip_data.sort_values('lower_bound_ip_address', inplace=True)

            logging.info("Merging datasets...")
            # Perform an asof merge (optimized for range-based matching)
            merged_data = pd.merge_asof(
                fraud_data,
                ip_data,
                left_on='ip_int',
                right_on='lower_bound_ip_address',
                direction='backward'
            )

            # Filter rows where ip_int is within the lower and upper bounds
            merged_data = merged_data[
                (merged_data['ip_int'] >= merged_data['lower_bound_ip_address']) &
                (merged_data['ip_int'] <= merged_data['upper_bound_ip_address'])
            ]

            # Drop unnecessary columns
            merged_data.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)

            logging.info("Successfully merged fraud data with geolocation information.")

            return merged_data

        except Exception as e:
            logging.error(f"Error merging fraud data with geolocation: {e}")
            return fraud_data
