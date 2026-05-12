# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:28:49 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

# 1. Read data
df = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')

# 2. Define which columns are categorical and need one-hot encoding
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

# 3. One-hot encoding for these categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
df_categorical = df[categorical_cols]
ohe.fit(df_categorical)

encoded_array = ohe.transform(df_categorical)
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names)

# 4. Drop original text columns and merge with numeric data
df_numerical = df.drop(columns=categorical_cols)
df_final = pd.concat([df_numerical, df_encoded], axis=1)

# 5. Define numeric base columns (excluding the categorical ones)
base_features = [
    'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]

# 6. Identify the one-hot columns by prefix
one_hot_prefixes = [
    'Photocatalyst category_',
    'Membrane materials_',
    'Membrane type_',
    'Light frequency_',
    'Hybrid methods_'
]
one_hot_cols = [
    col for col in df_final.columns
    if any(col.startswith(prefix) for prefix in one_hot_prefixes)
]

# 7. Combine final features
final_feature_list = base_features + one_hot_cols

# 8. Select features (X) and define binary classification target (y)
X = df_final[final_feature_list].values
y = (df_final['Removal efficiency'] > 0.5).astype(int).values

# -------------------------------------------------------------------
# IMPORTANT: Do NOT z-score ALL features here (avoids leakage).
# Only z-score numeric base columns inside each training fold.
# -------------------------------------------------------------------
num_start, num_end = 0, len(base_features)

# 10. Setup K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# 11. Initialize storage for results of each run
run_results = {
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'Balanced Accuracy': [],
    'AUROC': [],
    'F1 Score': []
}

rf_param_list = [
    # (A) original
    {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
    # (B) larger forest
    {'n_estimators': 300, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
    # (C) moderate forest with light regularization
    {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 2, 'min_samples_leaf': 2},
]

# Perform multiple runs of cross-validation
n_runs = 5

for run in range(n_runs):
    accuracies, sensitivities, specificities, balanced_accuracies, aurocs, f1_scores = [], [], [], [], [], []

    # IMPORTANT: split on raw X (not scaled) to avoid leakage
    for train_index, test_index in kf.split(X):

        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric columns (no leakage)
        # ---------------------------------------------------------
        train_features_raw = X[train_index].copy()
        test_features_raw  = X[test_index].copy()

        scaler_fold = StandardScaler()
        scaler_fold.fit(train_features_raw[:, num_start:num_end])  # fit on TRAIN numeric only

        train_features_full = train_features_raw.copy()
        test_features       = test_features_raw.copy()
        train_features_full[:, num_start:num_end] = scaler_fold.transform(train_features_full[:, num_start:num_end])
        test_features[:,     num_start:num_end]   = scaler_fold.transform(test_features[:,     num_start:num_end])
        # ---------------------------------------------------------

        train_targets_full, test_targets = y[train_index], y[test_index]

        # ----- 90/10 validation split inside the training fold -----
        cut = int(len(train_features_full) * 0.9)
        X_tr, y_tr   = train_features_full[:cut], train_targets_full[:cut]
        X_val, y_val = train_features_full[cut:],  train_targets_full[cut:]

        # ----- Hyperparameter selection on validation set (by AUROC; fallback to accuracy) -----
        best_score, best_params = -np.inf, None
        for p in rf_param_list:
            clf = RandomForestClassifier(**p)
            clf.fit(X_tr, y_tr)
            try:
                val_probs = clf.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, val_probs)
            except ValueError:
                score = accuracy_score(y_val, clf.predict(X_val))
            if score > best_score:
                best_score, best_params = score, p

        # ----- Retrain best model on FULL fold training (train + val) -----
        classifier = RandomForestClassifier(**best_params)
        classifier.fit(train_features_full, train_targets_full)

        # Predict on the held-out test fold
        predictions = classifier.predict(test_features)
        test_probabilities = classifier.predict_proba(test_features)[:, 1]

        # Metrics (unchanged)
        accuracy = accuracy_score(test_targets, predictions)
        sensitivity = recall_score(test_targets, predictions)
        specificity = calculate_specificity(test_targets, predictions)
        balanced_accuracy = (sensitivity + specificity) / 2
        auroc = roc_auc_score(test_targets, test_probabilities)
        f1 = f1_score(test_targets, predictions)

        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        balanced_accuracies.append(balanced_accuracy)
        aurocs.append(auroc)
        f1_scores.append(f1)

    # Store average metrics for the current run (unchanged)
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

# Calculate and print the overall average and standard deviation (unchanged)
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")
