import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import yaml

def load_vectorized_data(train_path: str = 'data/vectorized/train_data_vectorized.csv') -> pd.DataFrame:
    """
    Load vectorized data from a CSV file.
    """
    train_data = pd.read_csv(train_path)
    return train_data

def evaluation_data(train_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Split the training data into training and evaluation sets.
    """
    train_data, evaluation_data = train_test_split(train_data, test_size=0.2, random_state=42)
    X_eval = evaluation_data.drop(columns=['label'])
    y_eval = evaluation_data['label']
    return train_data, X_eval, y_eval

def load_params(param_path: str = 'params.yaml') -> tuple[str, float, int, int]:
    """
    Load model building parameters from a YAML file.
    """
    with open(param_path, 'r') as f:
        params = yaml.safe_load(f)
        eval_metric = params['model_building']['eval_metric']
        learning_rate = params['model_building']['learning_rate']
        n_estimators = params['model_building']['n_estimators']
        max_depth = params['model_building']['max_depth']
    return eval_metric, learning_rate, n_estimators, max_depth

def build_model(train_data: pd.DataFrame, eval_metric: str = None, learning_rate: float = None, n_estimators: int = None, max_depth: int = None) -> XGBClassifier:
    """
    Build and train the XGBoost model.
    """
    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']

    model = XGBClassifier(use_label_encoder=False, eval_metric=eval_metric, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    return model

def evaluate_model(model: XGBClassifier, X_eval: pd.DataFrame, y_eval: pd.Series) -> float:
    """
    Evaluate the model on the evaluation set.
    """
    accuracy = model.score(X_eval, y_eval)
    return accuracy

def save_model(model: XGBClassifier, model_path: str = 'models/xgb_model.pkl') -> None:
    """
    Save the trained model to a file.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def main():
    """Main function to execute model building and evaluation."""
    train_data = load_vectorized_data()
    train_data, X_eval, y_eval = evaluation_data(train_data=train_data)
    eval_metric, learning_rate, n_estimators, max_depth = load_params()
    model = build_model(train_data=train_data, eval_metric=eval_metric, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    accuracy = evaluate_model(model, X_eval, y_eval)
    save_model(model)

if __name__ == '__main__':
    main()