import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

def load_data(url: str, file_path: str = 'data.csv') -> pd.DataFrame:
    """
    Load data from a URL or local file path.

    Parameters:
    url (str): URL to download the data from.
    file_path (str): Local file path to save/load the data.

    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        data = pd.read_csv(url)
    else:
        data = pd.read_csv(file_path)
    
    print(f"Data loaded with shape: {data.shape}")
    return data

def load_params(param_path: str = 'params.yaml') -> float:
    """
    Load parameters from a YAML file.

    Parameters:
    param_path (str): Path to the YAML file.

    Returns:
    float: Test size parameter loaded from the YAML file.
    """
    with open(param_path, 'r') as f:
        params = yaml.safe_load(f)
        test_size = params['data_ingestion']['test_size']
    return test_size

def encode_data(data: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical variables.

    Parameters:
    data (pd.DataFrame): The raw data.
    test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    pd.DataFrame: Preprocessed data.
    """
    data.drop(columns=['tweet_id'], inplace=True)  # Drop unnecessary columns
    data = data[data['sentiment'].isin(['happiness', 'sadness'])]  # Filter for specific sentiments
    data['sentiment'] = data['sentiment'].replace({'happiness': 1, 'sadness': 0})  # Encode sentiments

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data

def create_local_copy(train_data: pd.DataFrame, test_data: pd.DataFrame, train_path: str = 'train_data.csv', test_path: str = 'test_data.csv') -> None:
    """
    Save the preprocessed data to local CSV files.

    Parameters:
    train_data (pd.DataFrame): Training data.
    test_data (pd.DataFrame): Testing data.
    train_path (str): File path to save training data.
    test_path (str): File path to save testing data.
    """
    
    data_path = os.path.join('data', 'raw')
    os.makedirs(data_path, exist_ok=True)
    train_data.to_csv(os.path.join(data_path, train_path), index=False)
    test_data.to_csv(os.path.join(data_path, test_path), index=False)


def main():
    """Main function to execute data ingestion process."""
    data = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    test_size = load_params()
    train_data, test_data = encode_data(data, test_size)
    create_local_copy(train_data, test_data)


if __name__ == "__main__":
    main()