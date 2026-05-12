# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:54:09 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# Function to generate ECFPs from SMILES
def smiles_to_ecfp(smiles, radius=2, n_bits=1024):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smiles)
    return np.array(gen.GetFingerprint(mol)) if mol is not None else np.zeros((n_bits,))

# 1. Read and prepare data
data = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')
data['ECFP'] = data['SMILES'].apply(smiles_to_ecfp)

# 2. Define categorical columns to be one-hot encoded
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

# 3. One-hot encode the categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
df_categorical = data[categorical_cols]
encoded_array = ohe.fit_transform(df_categorical)
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=data.index)

# 4. Drop original categorical columns and merge with encoded features
data = data.drop(columns=categorical_cols)
data = pd.concat([data, df_encoded], axis=1)

# 5. Define numeric base features (ensure column names match your dataset)
numeric_cols = ['logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time']

# 6. Extract ECFP, numeric, and one-hot encoded features
ecfp_features = np.vstack(data['ECFP'].values)
numeric_features = data[numeric_cols].values
one_hot_cols = list(encoded_feature_names)  # use the encoder's output names
one_hot_features = data[one_hot_cols].values

# 7. Combine all features horizontally
features = np.hstack((ecfp_features, numeric_features, one_hot_features))

# 8. Create multi-class targets using bins
bins = [0, 0.5, 0.75, 1]
labels = [0, 1, 2]
targets = pd.cut(data['Removal efficiency'], bins=bins, labels=labels, include_lowest=True).astype(int)
targets_categorical = to_categorical(targets)

# -------------------------------------------------------------------
# IMPORTANT: Do NOT standardize full `features` here (avoids leakage).
# Only z-score numeric columns inside each training fold.
# -------------------------------------------------------------------
ecfp_size = ecfp_features.shape[1]
num_start = ecfp_size
num_end = ecfp_size + len(numeric_cols)

# 10. Setup 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# 11. Define the neural network model architecture
nn_param_list = [
    # (A) original
    {'layers': [256, 128, 64], 'dropout': 0.20, 'batch_size': 32, 'epochs': 50},
    # (B) bigger network, more dropout, larger batch
    {'layers': [512, 256, 128], 'dropout': 0.30, 'batch_size': 64, 'epochs': 60},
    # (C) smaller network, lighter dropout, more epochs
    {'layers': [128, 64, 32],  'dropout': 0.15, 'batch_size': 32, 'epochs': 80},
]

# Parameterized builder (keeps your architecture style)
def create_model_with_params(input_dim, params):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(params['layers'][0], activation='relu'),
        Dropout(params['dropout']),
        Dense(params['layers'][1], activation='relu'),
        Dropout(params['dropout']),
        Dense(params['layers'][2], activation='relu'),
        Dropout(params['dropout']),
        Dense(len(np.unique(targets)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 12. Function to calculate sensitivity and specificity (for binary comparisons)
def calculate_sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # For multi-class problems, you may need to compute these per class.
    # Here we compute sensitivity and specificity for class 1 as an example.
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) != 0 else 0
    specificity = cm[0, 0] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) != 0 else 0
    return sensitivity, specificity

# 13. Initialize storage for run results
run_results = {
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'Balanced Accuracy': [],
    'AUROC': [],
    'F1 Score': []
}

# 14. Perform multiple runs with 10-fold cross-validation
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

        train_targets, test_targets = targets_categorical[train_index], targets_categorical[test_index]

        # ---- 90/10 validation split inside the training fold ----
        cut = int(len(train_features) * 0.9)
        X_tr, y_tr   = train_features[:cut], train_targets[:cut]
        X_val, y_val = train_features[cut:],  train_targets[cut:]

        # ---- Hyperparameter selection on validation set (by AUROC; fallback to accuracy) ----
        best_score, best_params = -np.inf, None
        for params in nn_param_list:
            m = create_model_with_params(X_tr.shape[1], params)
            m.fit(X_tr, y_tr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
            val_probs = m.predict(X_val, verbose=0)
            try:
                val_score = roc_auc_score(y_val, val_probs, multi_class='ovr')
            except ValueError:
                val_pred = np.argmax(val_probs, axis=1)
                val_true = np.argmax(y_val, axis=1)
                val_score = accuracy_score(val_true, val_pred)
            if val_score > best_score:
                best_score, best_params = val_score, params

        # ---- Retrain best model on FULL fold training (train + val) ----
        model = create_model_with_params(train_features.shape[1], best_params)
        model.fit(train_features, train_targets,
                  epochs=best_params['epochs'],
                  batch_size=best_params['batch_size'],
                  verbose=0)

        # Predict on the test set
        predictions = model.predict(test_features, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(test_targets, axis=1)

        # Calculate metrics (unchanged)
        accuracy = accuracy_score(true_labels, predicted_labels)
        sensitivity, specificity = calculate_sensitivity_specificity(true_labels, predicted_labels)
        balanced_accuracy = (sensitivity + specificity) / 2
        auroc = roc_auc_score(test_targets, predictions, multi_class='ovr')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        # Store metrics for this fold
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
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

    print(f"Run {run+1} - Accuracy: {np.mean(accuracies):.3f}, Sensitivity: {np.mean(sensitivities):.3f}, "
          f"Specificity: {np.mean(specificities):.3f}, Balanced Accuracy: {np.mean(balanced_accuracies):.3f}, "
          f"AUROC: {np.mean(aurocs):.3f}, F1 Score: {np.mean(f1_scores):.3f}")

# Calculate and print the overall average and standard deviation of the metrics over all runs (unchanged)
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")
