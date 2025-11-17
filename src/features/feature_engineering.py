import numpy as np
import pandas as pd
import os
import yaml

from sklearn.feature_extraction.text import TfidfVectorizer

# load the interim data from data/interim
def load_interim_data(train_path: str = 'data/interim/train_data_interim.csv', test_path: str = 'data/interim/test_data_interim.csv') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load interim data from CSV files.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def load_params(param_path: str = 'params.yaml') -> int:
    """
    Load parameters from a YAML file.
    """
    with open(param_path, 'r') as f:
        params = yaml.safe_load(f)
        max_features = params['feature_engineering']['max_features']
    return max_features

# vectorize the text data
def vectorize_text(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorize the text data using TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    train_data['content'] = train_data['content'].fillna('').astype(str)
    test_data['content'] = test_data['content'].fillna('').astype(str)

    X_train = vectorizer.fit_transform(train_data['content'])
    X_test = vectorizer.transform(test_data['content'])

    train_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
    test_df = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())

    train_df['label'] = train_data['sentiment'].values
    test_df['label'] = test_data['sentiment'].values

    return train_df, test_df

# save the processed data to data/processed
def save_processed_data(train_data: pd.DataFrame, test_data: pd.DataFrame, train_path: str = 'train_data_processed.csv', test_path: str = 'test_data_processed.csv') -> None:
    """
    Save the processed data to CSV files.
    """
    data_path = os.path.join('data', 'processed')
    os.makedirs(data_path, exist_ok=True)
    train_data.to_csv(os.path.join(data_path, train_path), index=False)
    test_data.to_csv(os.path.join(data_path, test_path), index=False)

def main():
    """Main function to execute feature engineering."""
    train_data, test_data = load_interim_data()
    max_features = load_params()
    processed_train_data, processed_test_data = vectorize_text(train_data, test_data, max_features)
    save_processed_data(processed_train_data, processed_test_data)


if __name__ == '__main__':
    main()