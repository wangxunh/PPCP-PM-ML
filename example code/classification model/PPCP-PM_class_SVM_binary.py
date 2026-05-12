# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:31:54 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

# 1. Read data
df = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')

# 2. Define categorical columns and numeric base columns
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

base_features = [
    'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]

# 3. One-hot encoding for the specified categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
df_categorical = df[categorical_cols]
ohe.fit(df_categorical)

encoded_array = ohe.transform(df_categorical)
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names)

# 4. Drop original text columns and merge with numeric data
df_numerical = df.drop(columns=categorical_cols)
df_final = pd.concat([df_numerical, df_encoded], axis=1)

# 5. Identify one-hot columns by prefix
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

# 6. Combine final features: base numeric + one-hot
final_feature_list = base_features + one_hot_cols

# 7. Select features (X) and define binary classification target (y)
X = df_final[final_feature_list].values
y = (df_final['Removal efficiency'] > 0.5).astype(int)

# -------------------------------------------------------------------
# IMPORTANT: Do NOT z-score ALL features here (avoids leakage).
# Only z-score numeric columns inside each training fold.
# -------------------------------------------------------------------
num_start, num_end = 0, len(base_features)

# 9. K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# 10. Initialize storage for results
run_results = {
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'Balanced Accuracy': [],
    'AUROC': [],
    'F1 Score': []
}

svm_param_list = [
    # (A) original
    {'kernel': 'linear', 'C': 1.0, 'probability': True},
    # (B) linear SVM with lighter regularization
    {'kernel': 'linear', 'C': 0.5, 'probability': True},
    # (C) RBF kernel
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True},
]

n_runs = 5
for run in range(n_runs):
    accuracies = []
    sensitivities = []
    specificities = []
    balanced_accuracies = []
    aurocs = []
    f1_scores = []

    # IMPORTANT: split on raw `X` (not scaled) to avoid leakage
    for train_index, test_index in kf.split(X):
        train_features_raw, test_features_raw = X[train_index].copy(), X[test_index].copy()
        train_targets, test_targets = y[train_index], y[test_index]

        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric columns (no leakage)
        # ---------------------------------------------------------
        scaler_fold = StandardScaler()
        scaler_fold.fit(train_features_raw[:, num_start:num_end])  # fit on TRAIN numeric only

        train_features = train_features_raw.copy()
        test_features  = test_features_raw.copy()
        train_features[:, num_start:num_end] = scaler_fold.transform(train_features[:, num_start:num_end])
        test_features[:,  num_start:num_end] = scaler_fold.transform(test_features[:,  num_start:num_end])
        # ---------------------------------------------------------

        # ---- 90/10 validation split inside the training fold ----
        cut = int(len(train_features) * 0.9)
        X_tr, y_tr   = train_features[:cut], train_targets[:cut]
        X_val, y_val = train_features[cut:],  train_targets[cut:]

        # ---- Hyperparameter selection on validation set (prefer AUROC; fallback to accuracy) ----
        best_score, best_params = -np.inf, None
        for p in svm_param_list:
            svm_val = SVC(**p)
            svm_val.fit(X_tr, y_tr)
            try:
                val_probs = svm_val.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, val_probs)
            except ValueError:
                score = accuracy_score(y_val, svm_val.predict(X_val))
            if score > best_score:
                best_score, best_params = score, p

        # ---- Retrain best model on FULL fold training (train + val) ----
        svm_classifier = SVC(**best_params)
        svm_classifier.fit(train_features, train_targets)

        # Predict on the test set (unchanged)
        predictions = svm_classifier.predict(test_features)
        test_probabilities = svm_classifier.predict_proba(test_features)[:, 1]

        # Calculate metrics (unchanged)
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

    run_results['Accuracy'].append(np.mean(accuracies))
    run_results['Sensitivity'].append(np.mean(sensitivities))
    run_results['Specificity'].append(np.mean(specificities))
    run_results['Balanced Accuracy'].append(np.mean(balanced_accuracies))
    run_results['AUROC'].append(np.mean(aurocs))
    run_results['F1 Score'].append(np.mean(f1_scores))

    print(f"Run {run+1} - Average Metrics:")
    print(f"  Accuracy: {np.mean(accuracies):.3f}")
    print(f"  Sensitivity: {np.mean(sensitivities):.3f}")
    print(f"  Specificity: {np.mean(specificities):.3f}")
    print(f"  Balanced Accuracy: {np.mean(balanced_accuracies):.3f}")
    print(f"  AUROC: {np.mean(aurocs):.3f}")
    print(f"  F1 Score: {np.mean(f1_scores):.3f}")

# 11. Calculate and print overall average and std of metrics
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")
