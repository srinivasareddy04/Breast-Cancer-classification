# Breast Cancer Classification

A machine learning project that classifies tumors as malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.

## Models Compared
- Logistic Regression
- Support Vector Machine (SVM, linear kernel)
- K-Nearest Neighbors (KNN, k=6)
- Random Forest (98 trees)

## Dataset
- 569 samples, 30 numeric features
- Target: 0 = malignant, 1 = benign
- Loaded with `sklearn.datasets.load_breast_cancer()`

## Workflow
1. Load the dataset
2. Split into 70% training and 30% testing (stratified)
3. Scale features with StandardScaler (for Logistic Regression, SVM, KNN)
4. Train each model and evaluate it with accuracy, precision, recall and F1 score
5. Compare all models in a table and a bar chart
6. Predict on a new sample

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python your_script_name.py
```

## Output
- Classification report for each model
- Final comparison table
- Bar chart of model accuracy
- Predictions for a new sample

## Technologies
Python, pandas, matplotlib, scikit-learn
