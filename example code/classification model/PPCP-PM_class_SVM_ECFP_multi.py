# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:29:09 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Function to generate ECFPs from SMILES
def smiles_to_ecfp(smiles, radius=2, n_bits=1024):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smiles)
    return np.array(gen.GetFingerprint(mol)) if mol is not None else np.zeros((n_bits,))

# Read and prepare data
data = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')
data['ECFP'] = data['SMILES'].apply(smiles_to_ecfp)

# Define numeric columns (ensure these column names match your dataset)
numeric_cols = ['logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time']

# Define raw categorical columns to be one-hot encoded.
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
other_features = np.hstack((features_numeric, df_encoded.values))

# Extract ECFP features (convert list of arrays to 2D NumPy array)
ecfp_features = np.array(data['ECFP'].tolist())

# Combine all features horizontally
features = np.hstack((ecfp_features, other_features))

# Define the target variable for multiclass classification
bins = [0, 0.5, 0.75, 1]
labels = [0, 1, 2]
targets = pd.cut(data['Removal efficiency'], bins=bins, labels=labels, include_lowest=True).astype(int)
targets_categorical = to_categorical(targets)

# -------------------------------------------------------------------
# IMPORTANT: Do NOT z-score ALL features here (avoids leakage).
# Only z-score numeric columns inside each training fold.
# -------------------------------------------------------------------
ecfp_size = ecfp_features.shape[1]
num_start = ecfp_size
num_end = ecfp_size + len(numeric_cols)

# Setup 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# Function to calculate sensitivity and specificity
def calculate_sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = []
    specificity = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = cm.sum() - (tp + fn + fp)
        sens = tp / (tp + fn) if (tp + fn) != 0 else 0
        spec = tn / (tn + fp) if (tn + fp) != 0 else 0
        sensitivity.append(sens)
        specificity.append(spec)
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

# Initialize SVM with probability estimates (kept, though it is overwritten later)
model = SVC(probability=True, kernel='rbf', C=1.0)

svm_param_list = [
    # (A) original
    {'kernel': 'linear', 'C': 1.0, 'probability': True},
    # (B) linear with lighter regularization
    {'kernel': 'linear', 'C': 0.5, 'probability': True},
    # (C) RBF kernel
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True},
]

# Perform multiple runs
n_runs = 5
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

        train_targets = np.argmax(targets_categorical[train_index], axis=1)
        test_targets  = np.argmax(targets_categorical[test_index],  axis=1)

        # ---- 90/10 validation split inside the training fold ----
        cut = int(len(train_features) * 0.9)
        X_tr, y_tr   = train_features[:cut], train_targets[:cut]
        X_val, y_val = train_features[cut:],  train_targets[cut:]

        # ---- Hyperparameter selection on validation set (prefer AUROC; fallback to accuracy) ----
        best_score, best_params = -np.inf, None
        for p in svm_param_list:
            model_val = SVC(**p)
            model_val.fit(X_tr, y_tr)
            try:
                val_probs = model_val.predict_proba(X_val)
                score = roc_auc_score(to_categorical(y_val, num_classes=len(labels)), val_probs, multi_class='ovr')
            except ValueError:
                score = accuracy_score(y_val, model_val.predict(X_val))
            if score > best_score:
                best_score, best_params = score, p

        # ---- Retrain best model on FULL fold training (train + val) ----
        model = SVC(**best_params)
        model.fit(train_features, train_targets)

        # Predict on the test set (unchanged)
        predictions = model.predict(test_features)
        probabilities = model.predict_proba(test_features)

        # Calculate metrics (unchanged)
        accuracy = accuracy_score(test_targets, predictions)
        sensitivity, spec = calculate_sensitivity_specificity(test_targets, predictions)
        balanced_accuracy = (sensitivity + spec) / 2
        auroc = roc_auc_score(to_categorical(test_targets, num_classes=len(labels)), probabilities, multi_class='ovr')
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

    print(f"Run {run+1} - Average Metrics:")
    print(f"  Accuracy: {np.mean(accuracies):.3f}")
    print(f"  Sensitivity: {np.mean(sensitivities):.3f}")
    print(f"  Specificity: {np.mean(specificities):.3f}")
    print(f"  Balanced Accuracy: {np.mean(balanced_accuracies):.3f}")
    print(f"  AUROC: {np.mean(aurocs):.3f}")
    print(f"  F1 Score: {np.mean(f1_scores):.3f}")

# Print the overall average and standard deviation of the metrics over all runs
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")
