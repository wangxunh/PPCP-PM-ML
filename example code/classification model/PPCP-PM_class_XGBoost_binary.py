# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:34:56 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

# 1. Read the CSV data
df = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')

# 2. Define categorical columns to be one-hot encoded
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

# 3. Initialize and fit the OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
df_categorical = df[categorical_cols]
ohe.fit(df_categorical)

# 4. Transform the data and create a DataFrame for the one-hot columns
encoded_array = ohe.transform(df_categorical)
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names)

# 5. Drop original text columns and merge them with the numeric data
df_numerical = df.drop(columns=categorical_cols)
df_final = pd.concat([df_numerical, df_encoded], axis=1)

# 6. Define numeric base columns (excluding the original categorical ones)
base_features = [
    'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]

# 7. Identify one-hot columns by prefix
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

# 8. Combine final feature list: numeric base + one-hot columns
final_feature_list = base_features + one_hot_cols

# 9. Select features (X) and define binary classification target (y)
X = df_final[final_feature_list].values
y = (df_final['Removal efficiency'] > 0.5).astype(int).values  # ensure numpy array

# -------------------------------------------------------------------
# IMPORTANT: Do NOT scale here (avoids leakage).
# Only z-score numeric features (base_features) inside each training fold.
# Numeric block is the first len(base_features) columns in `X`.
# -------------------------------------------------------------------
num_start, num_end = 0, len(base_features)

# 11. Setup K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# 12. Initialize storage for metrics
run_results = {
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'Balanced Accuracy': [],
    'AUROC': [],
    'F1 Score': []
}

xgb_param_list = [
    # (A) original
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8},
    # (B) deeper trees, slower learning, more estimators
    {'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 8, 'subsample': 0.9, 'colsample_bytree': 0.8},
    # (C) shallower trees, faster learning
    {'n_estimators': 300, 'learning_rate': 0.2, 'max_depth': 4, 'subsample': 0.7, 'colsample_bytree': 0.6},
]

n_runs = 5
run_results = {
    'Accuracy': [], 'Sensitivity': [], 'Specificity': [],
    'Balanced Accuracy': [], 'AUROC': [], 'F1 Score': []
}

for run in range(n_runs):
    accuracies = []
    sensitivities = []
    specificities = []
    balanced_accuracies = []
    aurocs = []
    f1_scores = []

    # IMPORTANT: split on raw X (not scaled) to avoid leakage
    for train_index, test_index in kf.split(X):
        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric base features
        # ---------------------------------------------------------
        train_features_raw = X[train_index].copy()
        test_features_raw  = X[test_index].copy()

        scaler_fold = StandardScaler()
        scaler_fold.fit(train_features_raw[:, num_start:num_end])  # fit on TRAIN numeric only

        train_features = train_features_raw.copy()
        test_features  = test_features_raw.copy()
        train_features[:, num_start:num_end] = scaler_fold.transform(train_features[:, num_start:num_end])
        test_features[:,  num_start:num_end] = scaler_fold.transform(test_features[:,  num_start:num_end])
        # ---------------------------------------------------------

        train_targets, test_targets = y[train_index], y[test_index]

        # ---- 90/10 validation split inside the training fold ----
        cut = int(len(train_features) * 0.9)
        X_tr, y_tr   = train_features[:cut], train_targets[:cut]
        X_val, y_val = train_features[cut:],  train_targets[cut:]

        # ---- Select best params on validation set (by AUROC; fallback to accuracy) ----
        best_score, best_params = -np.inf, None
        for p in xgb_param_list:
            classifier = xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='logloss', **p
            )
            classifier.fit(X_tr, y_tr)
            try:
                val_probs = classifier.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, val_probs)
            except ValueError:
                score = accuracy_score(y_val, classifier.predict(X_val))
            if score > best_score:
                best_score, best_params = score, p

        # ---- Retrain best model on FULL fold training (train + val) ----
        classifier = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', **best_params
        )
        classifier.fit(train_features, train_targets)

        # Predict on the test set (unchanged)
        predictions = classifier.predict(test_features)
        test_probabilities = classifier.predict_proba(test_features)[:, 1]

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

# 13. Calculate and print the overall average and standard deviation of metrics over all runs
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")
