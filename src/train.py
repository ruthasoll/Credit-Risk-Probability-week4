# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import os
import joblib

# Constants
DATA_PATH = "data/processed/processed_data.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42

def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found at {path}")
    return pd.read_csv(path)

def evaluate_model(y_test, y_pred, y_pred_proba):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }

def train_models():
    # Load Data
    df = load_data(DATA_PATH)
    
    # Prepare X and y
    # Drop non-feature columns
    drop_cols = ['CustomerId', 'is_high_risk']
    # If there are other non-numeric columns that weren't encoded/dropped, we should handle them.
    # The previous step (data_processing) kept 'most_common_...' cols but also added '_woe' cols.
    # Models generally need numeric input. We should use WOE cols or encoded cols.
    # The 'most_common_' cols are still strings. We should drop them and use '_woe' versions or numeric cols.
    
    X = df.drop(columns=drop_cols)
    y = df['is_high_risk']
    
    # Drop string columns (original categoricals)
    X = X.select_dtypes(include=[np.number])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # MLflow Setup
    mlflow.set_experiment("Credit_Risk_Model_Experiment")
    
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            "params": {'C': [0.1, 1, 10]}
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
        }
    }
    
    best_overall_model = None
    best_roc_auc = 0
    
    for name, config in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            grid = GridSearchCV(config["model"], config["params"], cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            
            # Log params and metrics
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, name)
            
            print(f"{name} Metrics: {metrics}")
            
            if metrics['roc_auc'] > best_roc_auc:
                best_roc_auc = metrics['roc_auc']
                best_overall_model = best_model
                
            # Save local artifact
            joblib.dump(best_model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    print(f"\nBest Model: {best_overall_model}")
    # Save best model finalized
    if best_overall_model:
        final_path = os.path.join(MODEL_DIR, "best_model.pkl")
        joblib.dump(best_overall_model, final_path)
        print(f"Best model saved to {final_path}")
        
        # Register in MLflow (Mocking this step as it usually requires a server)
        # mlflow.register_model(f"runs:/{run_id}/{name}", "CreditRiskModel")

if __name__ == "__main__":
    train_models()