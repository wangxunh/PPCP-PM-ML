# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:08:30 2024

@author: wangx
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import xgboost as xgb
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

# Define categorical columns to one-hot encode.
# (Assumes these columns are available in the original CSV file.)
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

# Define numeric columns (ensure these match your CSV)
numeric_cols = [
  'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]
numeric_features = data[numeric_cols].values

# Extract ECFP features (convert list of arrays to a 2D numpy array)
ecfp_features = np.vstack(data['ECFP'].values)

# Combine features: ECFP, numeric, and one-hot encoded categorical features
features = np.hstack((ecfp_features, numeric_features, categorical_features))

# Define the binary classification target
targets = (data['Removal efficiency'] > 0.5).astype(int).values

# -------------------------------------------------------------------
# IMPORTANT: Do NOT scale here (avoids leakage).
# Only z-score numeric columns (numeric_cols) inside each training fold.
# Numeric block starts right after ECFP and ends after numeric features.
# -------------------------------------------------------------------
ecfp_dim = ecfp_features.shape[1]
num_start = ecfp_dim
num_end = ecfp_dim + len(numeric_cols)
# -------------------------------------------------------------------

# Setup K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# Function to calculate specificity from the confusion matrix
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

# Initialize storage for results over multiple runs
run_results = {
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'Balanced Accuracy': [],
    'AUROC': [],
    'F1 Score': []
}

xgb_param_list = [
    # (A) defaults only
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8},
    # (B) more estimators, slower learning, deeper trees
    {'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 8, 'subsample': 0.9, 'colsample_bytree': 0.8},
    # (C) fewer/deeper constraints, faster learning
    {'n_estimators': 300, 'learning_rate': 0.2, 'max_depth': 4, 'subsample': 0.7, 'colsample_bytree': 0.6},
]

# Perform multiple runs
n_runs = 5
run_results = {
    'Accuracy': [], 'Sensitivity': [], 'Specificity': [],
    'Balanced Accuracy': [], 'AUROC': [], 'F1 Score': []
}

for run in range(n_runs):
    accuracies, sensitivities, specificities, balanced_accuracies, aurocs, f1_scores = [], [], [], [], [], []
    
    # IMPORTANT: split on raw `features` (not scaled) to avoid leakage
    for train_index, test_index in kf.split(features):
        # Fold train/test (raw)
        train_features_full_raw = features[train_index].copy()
        test_features_raw       = features[test_index].copy()
        train_targets_full, test_targets = targets[train_index], targets[test_index]
        
        # ---- Fold-safe scaling: z-score ONLY numeric part, fit on training fold ----
        scaler_fold = StandardScaler()
        scaler_fold.fit(train_features_full_raw[:, num_start:num_end])

        train_features_full = train_features_full_raw.copy()
        test_features       = test_features_raw.copy()
        train_features_full[:, num_start:num_end] = scaler_fold.transform(train_features_full[:, num_start:num_end])
        test_features[:,      num_start:num_end]  = scaler_fold.transform(test_features[:,      num_start:num_end])
        # -------------------------------------------------------------------------
        
        # ---- 90/10 validation split inside the training fold ----
        cut = int(len(train_features_full) * 0.9)
        X_tr, y_tr   = train_features_full[:cut], train_targets_full[:cut]
        X_val, y_val = train_features_full[cut:],  train_targets_full[cut:]
        
        # ---- Select best params on validation set (prefer AUROC; fallback to accuracy) ----
        best_score, best_params = -np.inf, None
        for p in xgb_param_list:
            classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **p)
            classifier.fit(X_tr, y_tr)
            try:
                val_probs = classifier.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, val_probs)
            except ValueError:
                score = accuracy_score(y_val, classifier.predict(X_val))
            if score > best_score:
                best_score, best_params = score, p

        # ---- Retrain best model on FULL fold training (train + val) ----
        classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **best_params)
        classifier.fit(train_features_full, train_targets_full)
        
        # Predict on the test set
        predictions = classifier.predict(test_features)
        test_probabilities = classifier.predict_proba(test_features)[:, 1]
        
        # Calculate evaluation metrics (unchanged)
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

# Calculate and print the overall average and standard deviation of the metrics over all runs (unchanged)
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")
