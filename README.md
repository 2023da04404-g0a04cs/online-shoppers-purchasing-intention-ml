# Online Shoppers Purchasing Intention — ML Classification

## a. Problem Statement

The goal of this project is to predict whether an online shopping session will result in a purchase (revenue generation) based on user browsing behavior, session characteristics, and temporal features. This is a **binary classification** problem where the target variable `Revenue` indicates whether the session ended with a purchase (`True`) or not (`False`).

Accurately predicting purchase intent enables e-commerce platforms to personalize user experience, optimize marketing campaigns, and increase conversion rates.

## b. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset  
**Source:** [UCI Machine Learning Repository (ID: 468)](https://archive.ics.uci.edu/dataset/468)  
**Citation:** Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository.

| Property | Value |
|----------|-------|
| **Instances** | 12,330 |
| **Features** | 17 (10 numerical + 7 categorical) |
| **Target** | Revenue (Binary: True/False) |
| **Class Distribution** | ~85.8% No Purchase, ~14.2% Purchase |

### Feature Descriptions

- **Administrative, Informational, ProductRelated** — Number of pages visited in each category during the session
- **Administrative_Duration, Informational_Duration, ProductRelated_Duration** — Total time spent on each page category
- **BounceRates** — Percentage of visitors who enter the site from that page and then leave without further interaction
- **ExitRates** — Percentage of pageviews that were the last in the session for that page
- **PageValues** — Average value of the page averaged over the value of the target page and/or completion of an eCommerce transaction
- **SpecialDay** — Closeness of the browsing date to a special day (e.g., Valentine's Day, Mother's Day)
- **Month** — Month of the year the session occurred
- **OperatingSystems** — Operating system of the visitor
- **Browser** — Browser used by the visitor
- **Region** — Geographic region of the visitor
- **TrafficType** — Traffic source type (e.g., direct, referral, search)
- **VisitorType** — New visitor, returning visitor, or other
- **Weekend** — Boolean indicating whether the session occurred on a weekend

## c. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8650 | 0.6525 | 0.6441 | 0.1086 | 0.1858 | 0.2253 |
| Decision Tree | 0.8414 | 0.5699 | 0.3277 | 0.1114 | 0.1663 | 0.1199 |
| kNN | 0.8597 | 0.5932 | 0.5385 | 0.0800 | 0.1393 | 0.1668 |
| Naive Bayes | 0.8329 | 0.6076 | 0.3524 | 0.2114 | 0.2643 | 0.1840 |
| Random Forest (Ensemble) | 0.8605 | 0.6330 | 0.5577 | 0.0829 | 0.1443 | 0.1749 |
| XGBoost (Ensemble) | 0.8593 | 0.6419 | 0.5246 | 0.0914 | 0.1557 | 0.1746 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves the **highest accuracy (0.865)** and **highest AUC (0.6525)** among all models. Its precision is the best at 0.6441, meaning when it predicts a purchase, it is correct ~64% of the time. However, recall is low (0.1086), indicating it misses most actual purchases. This conservative behavior is typical for linear models on imbalanced datasets — it learns to predict the majority class well but struggles with the minority class. The MCC of 0.2253 (best among all models) suggests it has the best overall balanced performance accounting for all quadrants of the confusion matrix. |
| Decision Tree | Shows the **lowest overall performance** with the lowest AUC (0.5699) and MCC (0.1199). Accuracy of 0.8414 is decent but misleading given the class imbalance. Low precision (0.3277) means many false positives, while low recall (0.1114) means it misses most purchases. The decision tree tends to overfit to majority class patterns and creates overly specific splits that don't generalize well. Without aggressive pruning or class balancing techniques, it struggles significantly on imbalanced data. |
| kNN | Achieves good accuracy (0.8597) but has the **lowest recall (0.08)** among all models, meaning it identifies very few actual purchases. Precision is moderate at 0.5385. The kNN algorithm is sensitive to class imbalance since majority class neighbors dominate the voting process. The distance-based approach also suffers in high-dimensional feature spaces. The low F1 (0.1393) and moderate MCC (0.1668) confirm poor minority class detection despite decent majority class performance. |
| Naive Bayes | Has the **highest recall (0.2114)** and **best F1 score (0.2643)** among all models, making it the most effective at detecting actual purchases. While accuracy is lowest at 0.8329, this is because it makes more positive predictions, trading accuracy for better minority class coverage. The independence assumption, though violated in practice, creates a model less biased toward the majority class. AUC of 0.6076 is moderate, and MCC of 0.184 reflects a reasonable balance. For applications where catching potential buyers matters more than avoiding false alarms, Naive Bayes is the strongest choice. |
| Random Forest (Ensemble) | Delivers the **second-highest precision (0.5577)** and good accuracy (0.8605), but recall is very low at 0.0829. The bagging ensemble averages predictions across 100 trees, which tends to produce conservative predictions favoring the majority class. Despite its reputation as a strong general-purpose classifier, the class imbalance limits its ability to identify purchases. AUC (0.633) is the second-best, suggesting the model has good discrimination ability that could be unlocked with techniques like class weighting or SMOTE. |
| XGBoost (Ensemble) | Performs similarly to Random Forest with accuracy of 0.8593 and precision of 0.5246, but slightly better recall (0.0914). AUC of 0.6419 is the **third-highest**, showing good ranking ability. As a boosting method, XGBoost sequentially corrects errors, which theoretically helps with minority class detection. However, without explicit class weight adjustment or threshold tuning, it still defaults to conservative predictions. The similar performance to Random Forest suggests the dataset's class imbalance is the primary challenge, not model complexity. |

### Overall Observations

1. **Class Imbalance Impact:** All models achieve 83-87% accuracy largely by correctly predicting "No Purchase" (the majority class). The more meaningful metrics — Recall, F1, and MCC — reveal significant room for improvement in identifying actual purchasers.

2. **Precision vs. Recall Trade-off:** Logistic Regression and kNN favor precision (fewer false positives), while Naive Bayes favors recall (catching more actual purchases). The choice depends on the business objective: minimize wasted marketing spend (high precision) or maximize captured purchases (high recall).

3. **Best Overall Model:** Considering all metrics holistically, **Logistic Regression** provides the best balanced performance (highest MCC: 0.2253), while **Naive Bayes** is best for maximizing purchase detection (highest F1: 0.2643 and recall: 0.2114).

4. **Potential Improvements:** Techniques such as SMOTE oversampling, class weights, threshold optimization, or cost-sensitive learning could significantly improve minority class detection across all models.

## Project Structure

```
project-folder/
│── app.py                  # Streamlit web application (trains models on-the-fly)
│── requirements.txt        # Python dependencies
│── README.md               # This file
│── model/
│   └── train_models.py     # Standalone model training script
```

> **Note:** The Streamlit app generates the dataset and trains all 6 models at startup using Streamlit's caching (`@st.cache_resource`), so no `.pkl` model files or `.csv` data files are needed in the repository. The `model/train_models.py` script is provided as a standalone reference for model training and evaluation.

## How to Run Locally

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Streamlit App Features

- **Model Selection Dropdown:** Choose from 6 trained classification models
- **Dataset Upload (CSV):** Upload test data for real-time prediction and evaluation
- **Evaluation Metrics Display:** Accuracy, AUC, Precision, Recall, F1, MCC for each model
- **Confusion Matrix:** Visual heatmap for each model's predictions
- **Classification Report:** Detailed per-class precision, recall, and F1-score
- **Model Comparison:** Side-by-side comparison chart of all models

## Deployment

Deployed on **Streamlit Community Cloud**.

**Live App Link:** *(Add after deployment)*

## Technologies Used

- Python 3.10+
- Scikit-learn
- XGBoost
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn

---
