# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:14:41 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
import xgboost as xgb
from tensorflow.keras.utils import to_categorical

# Read data
data = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')

# Define numeric columns (ensure these names match your dataset)
numeric_cols = ['logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time']

# Define categorical columns to be one-hot encoded.
# (Assumes the CSV contains the raw categorical values in these columns)
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

# One-hot encode the categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = ohe.fit_transform(data[categorical_cols])
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(cat_features, columns=encoded_feature_names, index=data.index)

# Combine numeric features and one-hot encoded categorical features
features_numeric = data[numeric_cols].values
features = np.hstack((features_numeric, df_encoded.values))

# Define the target variable for multiclass classification using bins
bins = [0, 0.5, 0.75, 1]
labels = [0, 1, 2]
targets = pd.cut(data['Removal efficiency'], bins=bins, labels=labels, include_lowest=True).astype(int)

# One-hot encode the target variable (for AUROC calculation)
targets_categorical = to_categorical(targets)

# -------------------------------------------------------------------
# IMPORTANT: Do NOT scale here (avoids leakage).
# Only z-score numeric columns inside each training fold.
# Numeric block is at the start of `features` (before one-hot columns).
# -------------------------------------------------------------------
num_start = 0
num_end = len(numeric_cols)
# -------------------------------------------------------------------

# Setup 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# Function to calculate sensitivity and specificity
def calculate_sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = np.diag(cm) / np.sum(cm, axis=1)
    specificity = np.diag(cm) / np.sum(cm, axis=0)
    return np.mean(sensitivity), np.mean(specificity)

# Initialize storage for results of each run
run_results = {
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'Balanced Accuracy': [],
    'AUROC': [],
    'F1 Score': []
}

# Perform multiple runs
xgb_param_list = [
    # (A) original (your current hyperparameters)
    {'objective': 'multi:softprob', 'num_class': len(labels)},
    # (B) more estimators, slower learning, deeper trees
    {'objective': 'multi:softprob', 'num_class': len(labels),
     'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 8,
     'subsample': 0.9, 'colsample_bytree': 0.8},
    # (C) faster learning, shallower trees
    {'objective': 'multi:softprob', 'num_class': len(labels),
     'n_estimators': 300, 'learning_rate': 0.2, 'max_depth': 4,
     'subsample': 0.7, 'colsample_bytree': 0.6},
]

# Perform multiple runs
n_runs = 5
for run in range(n_runs):
    accuracies, sensitivities, specificities, balanced_accuracies, aurocs, f1_scores = [], [], [], [], [], []
    
    # IMPORTANT: split on raw `features` (not scaled) to avoid leakage
    for train_index, test_index in kf.split(features):
        train_features_raw = features[train_index].copy()
        test_features_raw  = features[test_index].copy()
        train_targets = np.argmax(targets_categorical[train_index], axis=1)
        test_targets  = np.argmax(targets_categorical[test_index],  axis=1)

        # ---- Fold-safe scaling: z-score ONLY numeric part, fit on training fold ----
        scaler_fold = StandardScaler()
        scaler_fold.fit(train_features_raw[:, num_start:num_end])

        train_features = train_features_raw.copy()
        test_features  = test_features_raw.copy()
        train_features[:, num_start:num_end] = scaler_fold.transform(train_features[:, num_start:num_end])
        test_features[:,  num_start:num_end] = scaler_fold.transform(test_features[:,  num_start:num_end])
        # -------------------------------------------------------------------------

        # ---- 90/10 validation split inside the training fold ----
        cut = int(len(train_features) * 0.9)
        X_tr, y_tr   = train_features[:cut], train_targets[:cut]
        X_val, y_val = train_features[cut:],  train_targets[cut:]

        # ---- Select best params on validation set (prefer AUROC; fallback to accuracy) ----
        best_score, best_params = -np.inf, None
        for p in xgb_param_list:
            model_val = xgb.XGBClassifier(**p)
            model_val.fit(X_tr, y_tr)
            try:
                val_probs = model_val.predict_proba(X_val)
                score = roc_auc_score(to_categorical(y_val, num_classes=len(labels)),
                                      val_probs, multi_class='ovr')
            except ValueError:
                score = accuracy_score(y_val, model_val.predict(X_val))
            if score > best_score:
                best_score, best_params = score, p

        # ---- Retrain best model on FULL fold training (train + val) ----
        model = xgb.XGBClassifier(**best_params)
        model.fit(train_features, train_targets)

        # Predict on the test set (unchanged)
        predictions  = model.predict(test_features)
        probabilities = model.predict_proba(test_features)

        # Calculate metrics (unchanged)
        accuracy = accuracy_score(test_targets, predictions)
        sensitivity, spec = calculate_sensitivity_specificity(test_targets, predictions)
        balanced_accuracy = (sensitivity + spec) / 2
        auroc = roc_auc_score(to_categorical(test_targets, num_classes=len(labels)),
                              probabilities, multi_class='ovr')
        f1 = f1_score(test_targets, predictions, average='weighted')
        
        # Store metrics for this fold (unchanged)
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(spec)
        balanced_accuracies.append(balanced_accuracy)
        aurocs.append(auroc)
        f1_scores.append(f1)
    
    # Store average metrics for this run (unchanged)
    run_results['Accuracy'].append(np.mean(accuracies))
    run_results['Sensitivity'].append(np.mean(sensitivities))
    run_results['Specificity'].append(np.mean(specificities))
    run_results['Balanced Accuracy'].append(np.mean(balanced_accuracies))
    run_results['AUROC'].append(np.mean(aurocs))
    run_results['F1 Score'].append(np.mean(f1_scores))
    
    # Print average metrics for this run (unchanged)
    print(f"Run {run+1} - Average Metrics:")
    print(f"  Accuracy: {np.mean(accuracies):.3f}")
    print(f"  Sensitivity: {np.mean(sensitivities):.3f}")
    print(f"  Specificity: {np.mean(specificities):.3f}")
    print(f"  Balanced Accuracy: {np.mean(balanced_accuracies):.3f}")
    print(f"  AUROC: {np.mean(aurocs):.3f}")
    print(f"  F1 Score: {np.mean(f1_scores):.3f}")

# Calculate and print the overall average and standard deviation of the metrics over all runs (unchanged)
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")
