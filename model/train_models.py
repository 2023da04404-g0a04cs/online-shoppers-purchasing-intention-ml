"""
Online Shoppers Purchasing Intention - Classification Models
============================================================
Trains 6 classification models on the Online Shoppers Purchasing 
Intention Dataset and saves models along with evaluation metrics.

Dataset: UCI ML Repository (ID: 468)
Features: 17 (10 numerical + 7 categorical)
Target: Revenue (binary: True/False)
Instances: 12,330
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)


def load_and_preprocess_data():
    """Load the dataset from CSV and preprocess."""
    print("Loading dataset...")
    df = pd.read_csv("data/online_shoppers.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Revenue'].value_counts()}")
    return df


def preprocess_features(df):
    """Encode categorical variables and scale features."""
    df = df.copy()
    df = df.dropna()
    
    le_dict = {}
    categorical_cols = ['Month', 'VisitorType']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    return X_scaled, y, scaler, le_dict


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and evaluate them."""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'kNN': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=15, n_jobs=-1
        ),
        'XGBoost (Ensemble)': XGBClassifier(
            n_estimators=100, random_state=42, max_depth=6,
            learning_rate=0.1, use_label_encoder=False, eval_metric='logloss'
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'AUC': round(roc_auc_score(y_test, y_proba), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
        }
        
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report
        }
        trained_models[name] = model
        
        print(f"  Accuracy: {metrics['Accuracy']}")
        print(f"  AUC: {metrics['AUC']}")
        print(f"  Precision: {metrics['Precision']}")
        print(f"  Recall: {metrics['Recall']}")
        print(f"  F1: {metrics['F1']}")
        print(f"  MCC: {metrics['MCC']}")
    
    return results, trained_models


def save_artifacts(trained_models, results, scaler, le_dict, feature_names, X_test, y_test):
    """Save all models and artifacts."""
    os.makedirs("model", exist_ok=True)
    
    model_filenames = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'kNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest (Ensemble)': 'random_forest.pkl',
        'XGBoost (Ensemble)': 'xgboost.pkl'
    }
    
    for name, model in trained_models.items():
        filepath = os.path.join("model", model_filenames[name])
        joblib.dump(model, filepath)
        print(f"Saved {name} -> {filepath}")
    
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(le_dict, "model/label_encoders.pkl")
    
    with open("model/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("model/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    
    test_df = X_test.copy()
    test_df['Revenue'] = y_test.values
    test_df.to_csv("data/test_data.csv", index=False)
    
    print("\nAll artifacts saved successfully!")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    df = load_and_preprocess_data()
    X, y, scaler, le_dict = preprocess_features(df)
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    results, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    save_artifacts(trained_models, results, scaler, le_dict, feature_names, X_test, y_test)
    
    print("\n" + "="*90)
    print("MODEL COMPARISON TABLE")
    print("="*90)
    print(f"{'Model':<28} {'Accuracy':>10} {'AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}")
    print("-"*90)
    for name, res in results.items():
        m = res['metrics']
        print(f"{name:<28} {m['Accuracy']:>10.4f} {m['AUC']:>10.4f} {m['Precision']:>10.4f} {m['Recall']:>10.4f} {m['F1']:>10.4f} {m['MCC']:>10.4f}")
    print("="*90)
