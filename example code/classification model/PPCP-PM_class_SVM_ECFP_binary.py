# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:00:37 2024

@author: wangx
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Function to generate ECFPs from SMILES
def smiles_to_ecfp(smiles, radius=2, n_bits=1024):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smiles)
    return np.array(gen.GetFingerprint(mol)) if mol is not None else np.zeros((n_bits,))

# Read and prepare data
data = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')

# Generate ECFP features from the SMILES column
data['ECFP'] = data['SMILES'].apply(smiles_to_ecfp)

# Define categorical columns to one-hot encode
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

# One-hot encode the categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_features = ohe.fit_transform(data[categorical_cols])
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(categorical_features, columns=encoded_feature_names, index=data.index)

# Define numeric columns (ensure the column names match your CSV)
numeric_cols = [
    'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]
numeric_features = data[numeric_cols].values

# Extract ECFP features (convert list of arrays to 2D numpy array)
ecfp_features = np.vstack(data['ECFP'].values)

# Combine all features: ECFP, numeric, and one-hot encoded categorical features
features = np.hstack((ecfp_features, numeric_features, categorical_features))

# Define the binary classification target
y = (data['Removal efficiency'] > 0.5).astype(int).values

# NOTE:
# 1) Do NOT z-score all features.
# 2) Do NOT fit scaler on full dataset (avoid leakage).
# Only numeric columns will be z-scored INSIDE each CV fold.

# Setup K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# Indices for numeric block inside `features`
ecfp_dim = ecfp_features.shape[1]
num_start = ecfp_dim
num_end = ecfp_dim + len(numeric_cols)

# Function to calculate specificity
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

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
svm_param_list = [
    {'kernel': 'linear', 'C': 1.0, 'max_iter': 5000, 'cache_size': 1000},
    {'kernel': 'linear', 'C': 0.5, 'max_iter': 5000, 'cache_size': 1000},
    {'kernel': 'linear', 'C': 2.0, 'max_iter': 5000, 'cache_size': 1000},
]

n_runs = 5
for run in range(n_runs):
    accuracies = []
    sensitivities = []
    specificities = []
    balanced_accuracies = []
    aurocs = []
    f1_scores = []

    # IMPORTANT: split on raw `features` (not scaled) to avoid leakage
    for train_index, test_index in kf.split(features):

        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric columns (no leakage)
        # ---------------------------------------------------------
        X_train_raw = features[train_index].copy()
        X_test_raw  = features[test_index].copy()

        scaler_fold = StandardScaler()
        scaler_fold.fit(X_train_raw[:, num_start:num_end])  # fit on TRAIN numeric only

        X_train = X_train_raw.copy()
        X_test  = X_test_raw.copy()
        X_train[:, num_start:num_end] = scaler_fold.transform(X_train[:, num_start:num_end])
        X_test[:,  num_start:num_end] = scaler_fold.transform(X_test[:,  num_start:num_end])
        # ---------------------------------------------------------

        train_targets, test_targets = y[train_index], y[test_index]

        # ---- 90/10 validation split inside the training fold ----
        cut = int(len(X_train) * 0.9)
        X_tr, y_tr   = X_train[:cut], train_targets[:cut]
        X_val, y_val = X_train[cut:], train_targets[cut:]

        # ---- Hyperparameter selection on validation set (AUROC on decision scores) ----
        best_score, best_params = -np.inf, None
        for p in svm_param_list:
            svm_val = SVC(**{**p, 'probability': False})
            svm_val.fit(X_tr, y_tr)
            val_scores = svm_val.decision_function(X_val)
            score = roc_auc_score(y_val, val_scores)
            if score > best_score:
                best_score, best_params = score, p

        # ---- Retrain best model on FULL fold training (train + val) ----
        svm_classifier = SVC(**{**best_params, 'probability': False})
        svm_classifier.fit(X_train, train_targets)

        test_scores = svm_classifier.decision_function(X_test)
        auroc = roc_auc_score(test_targets, test_scores)

        # thresholded metrics (kept as in your code)
        predictions = (test_scores > 0).astype(int)

        # Calculate metrics (unchanged)
        accuracy = accuracy_score(test_targets, predictions)
        sensitivity = recall_score(test_targets, predictions)
        specificity = calculate_specificity(test_targets, predictions)
        balanced_accuracy = (sensitivity + specificity) / 2
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

# Calculate and print overall average and std of metrics
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")
