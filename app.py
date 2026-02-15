"""
Online Shoppers Purchasing Intention - Streamlit App
====================================================
Interactive ML Classification Dashboard
Author: Garimella Pavan Abhishek
Dataset: Online Shoppers Purchasing Intention (UCI ML Repository)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    """Load pre-computed model results."""
    with open("model/results.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_model(model_name):
    """Load a saved model."""
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'kNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest (Ensemble)': 'random_forest.pkl',
        'XGBoost (Ensemble)': 'xgboost.pkl'
    }
    return joblib.load(os.path.join("model", model_files[model_name]))

@st.cache_resource
def load_scaler_and_encoders():
    """Load the scaler and label encoders."""
    scaler = joblib.load("model/scaler.pkl")
    le_dict = joblib.load("model/label_encoders.pkl")
    return scaler, le_dict

@st.cache_data
def load_feature_names():
    """Load feature names."""
    with open("model/feature_names.json", "r") as f:
        return json.load(f)


def preprocess_uploaded_data(df, scaler, le_dict):
    """Preprocess uploaded CSV data for prediction."""
    df = df.copy()
    
    # Check if Revenue column exists (for evaluation)
    has_target = 'Revenue' in df.columns
    if has_target:
        y_true = df['Revenue'].astype(int)
        df = df.drop('Revenue', axis=1)
    else:
        y_true = None
    
    # Encode categorical columns
    categorical_cols = ['Month', 'VisitorType']
    for col in categorical_cols:
        if col in df.columns:
            le = le_dict[col]
            # Handle unseen labels gracefully
            known_labels = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_labels else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
    
    # Convert boolean columns
    if 'Weekend' in df.columns:
        df['Weekend'] = df['Weekend'].astype(int)
    
    # Ensure all expected features exist
    feature_names = load_feature_names()
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0
    
    df = df[feature_names]
    
    # Scale
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    
    return df_scaled, y_true


def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Purchase', 'Purchase'],
                yticklabels=['No Purchase', 'Purchase'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results):
    """Plot comparison bar chart of all models."""
    models = list(results.keys())
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    data = []
    for model in models:
        for metric in metrics:
            data.append({
                'Model': model,
                'Metric': metric,
                'Value': results[model]['metrics'][metric]
            })
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(models))
    width = 0.12
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4']
    
    for i, metric in enumerate(metrics):
        values = [results[m]['metrics'][metric] for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.85)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown('<p class="main-header">🛒 Online Shoppers Purchasing Intention</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML Classification Dashboard — 6 Models Compared</p>', unsafe_allow_html=True)
    
    # Load pre-computed results
    results = load_results()
    scaler, le_dict = load_scaler_and_encoders()
    
    # ─── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("Controls")
    
    # Model selection dropdown
    model_names = list(results.keys())
    selected_model = st.sidebar.selectbox(
        "🔍 Select Classification Model",
        model_names,
        index=0,
        help="Choose a model to view its detailed results"
    )
    
    st.sidebar.markdown("---")
    
    # CSV upload
    st.sidebar.subheader("Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file (test data)",
        type=['csv'],
        help="Upload test data CSV to evaluate the selected model. Include 'Revenue' column for evaluation metrics."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Dataset:** Online Shoppers Purchasing Intention  
    **Source:** UCI ML Repository  
    **Features:** 17  
    **Instances:** 12,330  
    **Task:** Binary Classification
    """)
    
    # ─── Tab Layout ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Comparison", 
        "Selected Model Details", 
        "Uploaded Data Results",
        "About"
    ])
    
    # ─── Tab 1: Model Comparison ──────────────────────────────────────────────
    with tab1:
        st.subheader("All Models — Evaluation Metrics Comparison")
        
        # Comparison table
        comparison_data = []
        for name, res in results.items():
            m = res['metrics']
            comparison_data.append({
                'Model': name,
                'Accuracy': m['Accuracy'],
                'AUC': m['AUC'],
                'Precision': m['Precision'],
                'Recall': m['Recall'],
                'F1 Score': m['F1'],
                'MCC': m['MCC']
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Style the dataframe
        styled_df = comp_df.style.highlight_max(
            subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'],
            color='#90EE90'
        ).format({
            'Accuracy': '{:.4f}',
            'AUC': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1 Score': '{:.4f}',
            'MCC': '{:.4f}'
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Bar chart comparison
        st.subheader("Visual Comparison")
        fig = plot_metrics_comparison(results)
        st.pyplot(fig)
        
        # Best model summary
        st.subheader("Key Observations")
        best_acc = max(results.items(), key=lambda x: x[1]['metrics']['Accuracy'])
        best_auc = max(results.items(), key=lambda x: x[1]['metrics']['AUC'])
        best_f1 = max(results.items(), key=lambda x: x[1]['metrics']['F1'])
        best_mcc = max(results.items(), key=lambda x: x[1]['metrics']['MCC'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Best Accuracy:** {best_acc[0]} ({best_acc[1]['metrics']['Accuracy']:.4f})")
            st.info(f"**Best AUC:** {best_auc[0]} ({best_auc[1]['metrics']['AUC']:.4f})")
        with col2:
            st.warning(f"**Best F1:** {best_f1[0]} ({best_f1[1]['metrics']['F1']:.4f})")
            st.error(f"**Best MCC:** {best_mcc[0]} ({best_mcc[1]['metrics']['MCC']:.4f})")
    
    # ─── Tab 2: Selected Model Details ────────────────────────────────────────
    with tab2:
        st.subheader(f"Detailed Results: {selected_model}")
        
        model_result = results[selected_model]
        metrics = model_result['metrics']
        
        # Metrics cards
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col2.metric("AUC", f"{metrics['AUC']:.4f}")
        col3.metric("Precision", f"{metrics['Precision']:.4f}")
        col4.metric("Recall", f"{metrics['Recall']:.4f}")
        col5.metric("F1 Score", f"{metrics['F1']:.4f}")
        col6.metric("MCC", f"{metrics['MCC']:.4f}")
        
        st.markdown("---")
        
        # Confusion Matrix
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Confusion Matrix")
            cm = np.array(model_result['confusion_matrix'])
            fig = plot_confusion_matrix(cm, selected_model)
            st.pyplot(fig)
        
        with col_right:
            st.subheader("Classification Report")
            report = model_result['classification_report']
            report_df = pd.DataFrame({
                'Class': ['No Purchase (0)', 'Purchase (1)', 'Macro Avg', 'Weighted Avg'],
                'Precision': [
                    report['0']['precision'],
                    report['1']['precision'],
                    report['macro avg']['precision'],
                    report['weighted avg']['precision']
                ],
                'Recall': [
                    report['0']['recall'],
                    report['1']['recall'],
                    report['macro avg']['recall'],
                    report['weighted avg']['recall']
                ],
                'F1-Score': [
                    report['0']['f1-score'],
                    report['1']['f1-score'],
                    report['macro avg']['f1-score'],
                    report['weighted avg']['f1-score']
                ],
                'Support': [
                    int(report['0']['support']),
                    int(report['1']['support']),
                    int(report['macro avg']['support']),
                    int(report['weighted avg']['support'])
                ]
            }).style.format({
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}'
            })
            st.dataframe(report_df, use_container_width=True, hide_index=True)
    
    # ─── Tab 3: Uploaded Data Results ─────────────────────────────────────────
    with tab3:
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.success(f"Uploaded file: {uploaded_file.name} ({uploaded_df.shape[0]} rows, {uploaded_df.shape[1]} columns)")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(uploaded_df.head(10), use_container_width=True)
                
                # Preprocess and predict
                model = load_model(selected_model)
                X_processed, y_true = preprocess_uploaded_data(uploaded_df, scaler, le_dict)
                
                y_pred = model.predict(X_processed)
                y_proba = model.predict_proba(X_processed)[:, 1]
                
                # Show predictions
                result_df = uploaded_df.copy()
                result_df['Predicted'] = y_pred
                result_df['Probability'] = np.round(y_proba, 4)
                
                st.subheader(f"Predictions using {selected_model}")
                st.dataframe(result_df.head(20), use_container_width=True)
                
                # If target column exists, show evaluation metrics
                if y_true is not None:
                    st.subheader(f"Evaluation Metrics on Uploaded Data")
                    
                    eval_metrics = {
                        'Accuracy': accuracy_score(y_true, y_pred),
                        'AUC': roc_auc_score(y_true, y_proba),
                        'Precision': precision_score(y_true, y_pred, zero_division=0),
                        'Recall': recall_score(y_true, y_pred, zero_division=0),
                        'F1': f1_score(y_true, y_pred, zero_division=0),
                        'MCC': matthews_corrcoef(y_true, y_pred)
                    }
                    
                    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                    mc1.metric("Accuracy", f"{eval_metrics['Accuracy']:.4f}")
                    mc2.metric("AUC", f"{eval_metrics['AUC']:.4f}")
                    mc3.metric("Precision", f"{eval_metrics['Precision']:.4f}")
                    mc4.metric("Recall", f"{eval_metrics['Recall']:.4f}")
                    mc5.metric("F1 Score", f"{eval_metrics['F1']:.4f}")
                    mc6.metric("MCC", f"{eval_metrics['MCC']:.4f}")
                    
                    # Confusion matrix for uploaded data
                    cm_uploaded = confusion_matrix(y_true, y_pred)
                    fig = plot_confusion_matrix(cm_uploaded, f"{selected_model} (Uploaded Data)")
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.subheader("Classification Report")
                    report_text = classification_report(y_true, y_pred, target_names=['No Purchase', 'Purchase'])
                    st.text(report_text)
                else:
                    st.info("Include a 'Revenue' column in your CSV to see evaluation metrics.")
                    
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
        else:
            st.info("Upload a CSV file from the sidebar to see predictions and evaluation results here.")
            st.markdown("""
            **Instructions:**
            1. Select a model from the sidebar dropdown
            2. Upload a CSV test file (with the same feature columns as the training data)
            3. Include a `Revenue` column (True/False or 1/0) to see evaluation metrics
            4. Results will appear here automatically
            """)
    
    # ─── Tab 4: About ─────────────────────────────────────────────────────────
    with tab4:
        st.subheader("About This Project")
        st.markdown("""
        ### Problem Statement
        Predict whether an online shopping session will result in a purchase (revenue generation) 
        based on user browsing behavior, session characteristics, and temporal features.
        
        ### Dataset Description
        The **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository 
        contains 12,330 sessions with 17 features (10 numerical, 7 categorical). The target variable 
        `Revenue` indicates whether the session ended with a purchase. The dataset is imbalanced with 
        approximately 85.8% negative (no purchase) and 14.2% positive (purchase) samples.
        
        ### Features
        - **Page Metrics:** Administrative, Informational, Product Related (counts and duration)
        - **Google Analytics:** Bounce Rate, Exit Rate, Page Values
        - **Temporal:** Month, Special Day, Weekend
        - **User Info:** Operating System, Browser, Region, Traffic Type, Visitor Type
        
        ### Models Implemented
        1. **Logistic Regression** — Linear model for binary classification
        2. **Decision Tree** — Tree-based model with interpretable splits
        3. **K-Nearest Neighbors (kNN)** — Instance-based lazy learner
        4. **Naive Bayes (Gaussian)** — Probabilistic classifier assuming feature independence
        5. **Random Forest (Ensemble)** — Bagging ensemble of decision trees
        6. **XGBoost (Ensemble)** — Gradient boosting ensemble method
        
        ### Tools & Technologies
        Python, Scikit-learn, XGBoost, Streamlit, Pandas, NumPy, Matplotlib, Seaborn
        """)
        
        st.markdown("---")


if __name__ == "__main__":
    main()
