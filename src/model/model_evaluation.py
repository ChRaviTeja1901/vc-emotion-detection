import numpy as np
import pandas as pd
import os
import json
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score

def load_model(model_path: str = 'models/xgb_model.pkl') -> XGBClassifier:
    """
    Load the trained model from a file.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_test_data(test_path: str = 'data/vectorized/test_data_vectorized.csv') -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the vectorized test data from a CSV file.
    """
    test_data = pd.read_csv(test_path)
    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']
    return X_test, y_test

def evaluate_model_on_data(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float, float, float]:
    """
    Evaluate the model on the test data.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy, precision, recall, roc_auc

def dump_evaluation_results(results: dict, result_path: str = 'results/evaluation_metrics.json') -> None:
    """
    Save the evaluation results to a JSON file.
    """
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)


def main():
    """Main function to execute model evaluation."""
    model = load_model()
    X_test, y_test = load_test_data()
    test_accuracy, test_precision, test_recall, test_roc_auc = evaluate_model_on_data(model, X_test, y_test)
    results = {
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test ROC AUC': test_roc_auc
    }
    dump_evaluation_results(results)

if __name__ == '__main__':
    main()