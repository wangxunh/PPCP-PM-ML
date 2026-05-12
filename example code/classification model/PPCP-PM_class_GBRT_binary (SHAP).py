# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:05:22 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import shap

# 1. Read data
data = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')

# 2. Define numeric columns (ensure these match your CSV)
numeric_cols = [
        'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]

# 3. Define raw categorical columns to be one-hot encoded
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

# 4. One-hot encode the categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = ohe.fit_transform(data[categorical_cols])
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)

# 5. Combine numeric features and one-hot encoded features
features_numeric = data[numeric_cols].values
features = np.hstack((features_numeric, cat_features))

# 6. Define binary classification target
targets = (data['Removal efficiency'] > 0.5).astype(int).values

# 7. NOTE: Do NOT standardize the full matrix here (avoids leakage).
#    Only numeric columns (the first len(numeric_cols) columns) will be z-scored inside each training fold.

# 8. Setup K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# numeric block indices inside `features` (numeric first, then one-hot)
num_start, num_end = 0, len(numeric_cols)

# 9. Initialize storage for each run
run_results = {
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'Balanced Accuracy': [],
    'AUROC': [],
    'F1 Score': []
}

# 10. Define a function to calculate specificity
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

# 11. Perform multiple runs
gbr_param_list = [
    # (A) defaults only (matches GradientBoostingClassifier() in your code)
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
    # (B) more trees, slower learning, same depth
    {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 3},
    # (C) fewer trees, faster learning, slightly deeper
    {'n_estimators': 150, 'learning_rate': 0.20, 'max_depth': 4},
]

n_runs = 5
best_params_last = None  # keep last chosen params to reuse for full-data fit at end

for run in range(n_runs):
    accuracies, sensitivities, specificities, balanced_accuracies, aurocs, f1_scores = [], [], [], [], [], []

    # IMPORTANT: split on raw `features` (not scaled) to avoid leakage
    for train_index, test_index in kf.split(features):

        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric columns (no leakage)
        # ---------------------------------------------------------
        train_features_raw = features[train_index].copy()
        test_features_raw  = features[test_index].copy()

        scaler_fold = StandardScaler()
        scaler_fold.fit(train_features_raw[:, num_start:num_end])  # fit on TRAIN numeric only

        train_features = train_features_raw.copy()
        test_features  = test_features_raw.copy()
        train_features[:, num_start:num_end] = scaler_fold.transform(train_features[:, num_start:num_end])
        test_features[:,  num_start:num_end] = scaler_fold.transform(test_features[:,  num_start:num_end])
        # ---------------------------------------------------------

        train_targets, test_targets = targets[train_index], targets[test_index]

        # ---- 90/10 validation split inside the training fold ----
        cut = int(len(train_features) * 0.9)
        X_tr, y_tr   = train_features[:cut], train_targets[:cut]
        X_val, y_val = train_features[cut:],  train_targets[cut:]

        # ---- Hyperparameter selection on validation set (prefer AUROC; fallback to accuracy) ----
        best_score, best_params = -np.inf, None
        for p in gbr_param_list:
            clf_val = GradientBoostingClassifier(**p)
            clf_val.fit(X_tr, y_tr)
            try:
                val_probs = clf_val.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, val_probs)
            except ValueError:
                score = accuracy_score(y_val, clf_val.predict(X_val))
            if score > best_score:
                best_score, best_params = score, p

        best_params_last = best_params  # store last seen for later full-data fit

        # ---- Retrain best model on FULL fold training (train + val) ----
        classifier = GradientBoostingClassifier(**best_params)
        classifier.fit(train_features, train_targets)

        # Predict on the test set
        predictions = classifier.predict(test_features)
        test_probabilities = classifier.predict_proba(test_features)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(test_targets, predictions)
        sensitivity = recall_score(test_targets, predictions)
        specificity = calculate_specificity(test_targets, predictions)
        balanced_accuracy = (sensitivity + specificity) / 2
        auroc = roc_auc_score(test_targets, test_probabilities)
        f1 = f1_score(test_targets, predictions)

        # Store metrics for this fold
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        balanced_accuracies.append(balanced_accuracy)
        aurocs.append(auroc)
        f1_scores.append(f1)

    # Store average metrics for this run
    run_results['Accuracy'].append(np.mean(accuracies))
    run_results['Sensitivity'].append(np.mean(sensitivities))
    run_results['Specificity'].append(np.mean(specificities))
    run_results['Balanced Accuracy'].append(np.mean(balanced_accuracies))
    run_results['AUROC'].append(np.mean(aurocs))
    run_results['F1 Score'].append(np.mean(f1_scores))

    # Print average metrics for this run
    print(f"Run {run+1} - Average Metrics:")
    print(f"  Accuracy: {np.mean(accuracies):.3f}")
    print(f"  Sensitivity: {np.mean(sensitivities):.3f}")
    print(f"  Specificity: {np.mean(specificities):.3f}")
    print(f"  Balanced Accuracy: {np.mean(balanced_accuracies):.3f}")
    print(f"  AUROC: {np.mean(aurocs):.4f}")
    print(f"  F1 Score: {np.mean(f1_scores):.3f}")

# ---------------------------
# 11. Print overall average and std
# ---------------------------
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")

# -------------------------------------------------------------------
# Feature Importance Section
# Train the classifier on the entire dataset to ensure it is fit
# -------------------------------------------------------------------
# Fit scaler on ALL numeric values here (no CV anymore; this is for interpretation plots on full data)
scaler_full = StandardScaler()
scaler_full.fit(features[:, num_start:num_end])

features_scaled_full = features.copy()
features_scaled_full[:, num_start:num_end] = scaler_full.transform(features_scaled_full[:, num_start:num_end])

# Use last selected hyperparameters (minimal change to your structure)
if best_params_last is None:
    best_params_last = gbr_param_list[0]

classifier = GradientBoostingClassifier(**best_params_last)
classifier.fit(features_scaled_full, targets)

# 1. SHAP Explainer & Values
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(features_scaled_full)

# 2. Combine numeric and encoded feature names
feature_names = numeric_cols + list(encoded_feature_names)

# -----------------------------
# A. Top-20 Chart (Aggregated by Raw Column)
# -----------------------------

# A1. Build a map from each one-hot encoded column to its raw column
subfeature_map = {}
for raw_col in categorical_cols:
    for enc_col in encoded_feature_names:
        # e.g., "Hybrid methods_dip-coating" -> "Hybrid methods"
        if enc_col.startswith(raw_col + "_"):
            subfeature_map[enc_col] = raw_col
        elif enc_col == raw_col:
            subfeature_map[enc_col] = raw_col

# A2. unify_feature_name: returns the raw column name if sub-feature
def unify_feature_name(name):
    if name in subfeature_map:
        return subfeature_map[name]
    else:
        return name

# A3. For each column in feature_names, unify if it's a sub-feature
unified_feature_names = [unify_feature_name(fn) for fn in feature_names]

# A4. Sum absolute SHAP values by raw column
abs_shap_values = np.abs(shap_values)
raw_feature_aggregation = {}
for i, unified_name in enumerate(unified_feature_names):
    col_mean = abs_shap_values[:, i].mean()  # mean absolute shap for sub-feature i
    if unified_name not in raw_feature_aggregation:
        raw_feature_aggregation[unified_name] = 0.0
    raw_feature_aggregation[unified_name] += col_mean

# A5. Build a DataFrame & plot the top 20
shap_df_agg = pd.DataFrame({
    'Feature': list(raw_feature_aggregation.keys()),
    'Mean Absolute SHAP Value': list(raw_feature_aggregation.values())
}).sort_values('Mean Absolute SHAP Value', ascending=False)

plt.figure(figsize=(10, 8))
top_k = 10
plt.barh(shap_df_agg['Feature'].head(top_k), shap_df_agg['Mean Absolute SHAP Value'].head(top_k))
plt.xlabel('Mean Absolute SHAP Value', fontsize=16)
plt.title('Top 10 Most Important Features - GBRT Binary Classification', fontsize=16)
plt.gca().invert_yaxis()  # largest bar on top
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# -----------------------------
# B. Grouped Feature Importance
# -----------------------------
group_names = {
    'Photocatalyst Properties': [
       'Photocatalyst category'
    ],
    'PPCP Properties': ['logP', 'MW'],
    'Membrane Properties': ['Membrane materials', 'Membrane type'],
    'Operational Conditions': [
        'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time', 'Light frequency', 'Hybrid methods'
    ]
}

grouped_shap_values = {}
for group, feats in group_names.items():
    # Gather indices in shap_values that unify to any raw feature in feats
    indices = []
    for i, ufn in enumerate(unified_feature_names):
        if ufn in feats:
            indices.append(i)

    # Sum absolute shap across sub-features, then average across samples
    if indices:
        sum_per_sample = abs_shap_values[:, indices].sum(axis=1)
        group_mean = sum_per_sample.mean()
    else:
        group_mean = 0.0

    grouped_shap_values[group] = group_mean

grouped_shap_df = pd.DataFrame({
    'Group': list(grouped_shap_values.keys()),
    'Mean Absolute SHAP Value': list(grouped_shap_values.values())
}).sort_values('Mean Absolute SHAP Value', ascending=False)

plt.figure(figsize=(10, 5))
plt.barh(grouped_shap_df['Group'], grouped_shap_df['Mean Absolute SHAP Value'], color='C0')
plt.xlabel('Mean Absolute SHAP Value', fontsize=16)
plt.title('GBRT Binary Classification', fontsize=16)
plt.gca().invert_yaxis()  # largest bar on top
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
