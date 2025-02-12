import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataVisualizer:
    """
    A class to handle univariate and bivariate data visualizations.
    """

    @staticmethod
    def plot_numerical_distribution(data, numerical_columns, title_prefix=""):
        """
        Plots the distribution of numerical features in the dataset.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing the features.
        numerical_columns : list
            List of numerical column names to plot.
        title_prefix : str
            Prefix to append to the plot title.
        """
        num_plots = len(numerical_columns)
        num_cols = 2
        num_rows = (num_plots + 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows))
        axes = axes.flatten()

        for i, feature in enumerate(numerical_columns):
            sns.histplot(data[feature], kde=True, bins=30, ax=axes[i], color=sns.color_palette('viridis', num_plots)[i])
            axes[i].set_title(f'{title_prefix}Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_class_distribution(data, target_column, title="Class Distribution"):
        """
        Plots the distribution of the target column.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing the target column.
        target_column : str
            The name of the target column.
        title : str
            Title of the plot.
        """
        plt.figure(figsize=(6, 4))
        sns.countplot(data=data, x=target_column, hue=target_column, palette='viridis', legend=False)
        plt.title(title)
        plt.xlabel(target_column)
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def plot_categorical_distribution(data, categorical_columns, title_prefix=""):
        """
        Plots the distribution of categorical features in the dataset.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing the features.
        categorical_columns : list
            List of categorical column names to plot.
        title_prefix : str
            Prefix to append to the plot title.
        """
        num_plots = len(categorical_columns)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 4 * num_rows))
        axes = axes.flatten()

        for i, feature in enumerate(categorical_columns):
            sns.countplot(data=data, x=feature, hue=feature , order=data[feature].value_counts().index, ax=axes[i], palette='viridis', legend=False)
            axes[i].set_title(f'{title_prefix}Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_boxplots_by_class(data, numerical_columns, target_column, palette="coolwarm"):
        """
        Plots boxplots for numerical features against the target column.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing the features.
        numerical_columns : list
            List of numerical column names to plot.
        target_column : str
            The target column to analyze against.
        palette : str
            Color palette for the plots.
        """
        num_plots = len(numerical_columns)
        num_cols = 2
        num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows))
        axes = axes.flatten()

        for i, feature in enumerate(numerical_columns):
            sns.boxplot(data=data, x=target_column, y=feature, hue=target_column, palette=palette, ax=axes[i])
            axes[i].set_title(f'{feature.capitalize()} by {target_column.capitalize()}')
            axes[i].set_xlabel(target_column.capitalize())
            axes[i].set_ylabel(feature.capitalize())

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_categorical_by_class(data, categorical_columns, target_column, palette="viridis"):
        """
        Plots count plots for categorical features against the target column.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing the features.
        categorical_columns : list
            List of categorical column names to plot.
        target_column : str
            The target column to analyze against.
        palette : str
            Color palette for the plots.
        """
        num_plots = len(categorical_columns)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 4 * num_rows))
        axes = axes.flatten()

        for i, feature in enumerate(categorical_columns):
            sns.countplot(data=data, x=feature, hue=target_column, palette=palette, ax=axes[i])
            axes[i].set_title(f'{feature.capitalize()} Distribution by {target_column.capitalize()}')
            axes[i].set_xlabel(feature.capitalize())
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(data, columns, title="Correlation Heatmap"):
        """
        Plots a heatmap of the correlation matrix.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing the features.
        columns : list
            List of numerical columns to include in the heatmap.
        title : str
            Title of the heatmap.
        """
        plt.figure(figsize=(10, 6))
        correlation_matrix = data[columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_temporal_patterns(data, time_columns):
        """
        Plots transaction counts over temporal features such as 'hour_of_day' and 'day_of_week'.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing temporal features.
        time_columns : list
            List of time-based feature column names (e.g., ['hour_of_day', 'day_of_week']).
        """
        titles = {'hour_of_day': 'Transactions by Hour of the Day',
                  'day_of_week': 'Transactions by Day of the Week'}
        x_labels = {'hour_of_day': 'Hour of the Day', 'day_of_week': 'Day of the Week (0 = Monday, 6 = Sunday)'}

        for col in time_columns:
            if col in data.columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=data, x=col, hue=col, palette='coolwarm' if col == 'hour_of_day' else 'viridis', legend=False)
                plt.title(titles[col])
                plt.xlabel(x_labels[col])
                plt.ylabel('Count')
                plt.show()

    @staticmethod
    def plot_transaction_distribution(data, column, title):
        """
        Plots the distribution of transaction-related features.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing the feature.
        column : str
            The column name to plot.
        title : str
            The title of the plot.
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], bins=30, kde=True)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def plot_top_users_by_transactions(data, user_id_column, transaction_column, top_n=20):
        """
        Plots the top users with the highest number of transactions.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing transaction data.
        user_id_column : str
            The column representing the user ID.
        transaction_column : str
            The column representing the transaction count.
        top_n : int
            Number of top users to display.
        """
        top_users = data.groupby(user_id_column)[transaction_column].max().nlargest(top_n)
        
        plt.figure(figsize=(12, 5))
        sns.barplot(x=top_users.index, hue= top_users.index, y=top_users.values, palette='coolwarm')
        plt.xticks(rotation=45)
        plt.title(f'Top {top_n} Users with Highest Transaction Counts')
        plt.xlabel(user_id_column)
        plt.ylabel(transaction_column)
        plt.show()


    @staticmethod
    def plot_fraud_rate_by_country(data, country_column='country', class_column='class', top_n=10):
        """
        Plots the fraud rate per country.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing country and fraud class data.
        country_column : str
            The column representing the country.
        class_column : str
            The column representing the fraud class (0 = Non-Fraud, 1 = Fraud).
        top_n : int
            Number of top fraudulent countries to display.
        """
        try:
            # Calculate fraud rate per country
            fraud_counts = data.groupby(country_column)[class_column].sum()
            total_counts = data[country_column].value_counts()
            fraud_rate = (fraud_counts / total_counts).fillna(0) * 100  # Convert to percentage

            # Select top N fraudulent countries
            top_fraud_countries = fraud_rate.nlargest(top_n)
            
            # Plot fraud rate by country
            plt.figure(figsize=(12, 6))
            sns.barplot(x=top_fraud_countries.index, hue= top_fraud_countries.index, y=top_fraud_countries.values, palette='Reds_r')
            plt.xticks(rotation=45)
            plt.title(f'Top {top_n} Countries by Fraud Rate')
            plt.xlabel('Country')
            plt.ylabel('Fraud Rate (%)')
            plt.show()

            logging.info(f"Plotted fraud rate for top {top_n} fraudulent countries.")

        except Exception as e:
            logging.error(f"Error in plotting fraud rate by country: {e}")

