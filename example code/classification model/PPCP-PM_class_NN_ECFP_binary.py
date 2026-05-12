# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:57:13 2024

@author: wangx
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Function to generate ECFPs from SMILES
def smiles_to_ecfp(smiles, radius=2, n_bits=1024):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smiles)
    return np.array(gen.GetFingerprint(mol)) if mol is not None else np.zeros((n_bits,))

# 1. Read data (update file name if needed)
df = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')

# 2. Generate ECFP features from SMILES column
df['ECFP'] = df['SMILES'].apply(smiles_to_ecfp)

# 3. Define categorical columns to be one-hot encoded
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

# 4. One-hot encode the categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
df_categorical = df[categorical_cols]
ohe.fit(df_categorical)
encoded_array = ohe.transform(df_categorical)
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=df.index)

# 5. Drop the original categorical columns and merge with the encoded data
df = df.drop(columns=categorical_cols)
df = pd.concat([df, df_encoded], axis=1)

# 6. Define numeric base features (ensure column names match your dataset)
base_features = [
    'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]

# 7. Identify one-hot encoded columns by prefix
one_hot_prefixes = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]
one_hot_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in one_hot_prefixes)]

# 8. Prepare the final feature set:
#    - ECFP features (stored as arrays in the 'ECFP' column)
#    - Numeric base features from the dataset
#    - One-hot encoded features
ecfp_features = np.vstack(df['ECFP'].values)
numeric_features = df[base_features].values
one_hot_features = df[one_hot_cols].values

# Combine all features horizontally
features = np.hstack((ecfp_features, numeric_features, one_hot_features))

# 9. Define the binary classification target
targets = (df['Removal efficiency'] > 0.5).astype(int).values

# -------------------------------------------------------------------
# IMPORTANT: Do NOT standardize full `features` here (avoids leakage).
# Only z-score numeric base columns inside each training fold.
# -------------------------------------------------------------------
ecfp_size = ecfp_features.shape[1]
num_start = ecfp_size
num_end = ecfp_size + len(base_features)

# 11. Setup 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# --- Hyperparameter sets (include your original as Option A) ---
nn_param_list = [
    # (A) original
    {'layers': [256, 128, 64], 'dropout': 0.20, 'learning_rate': 1e-3, 'batch_size': 32, 'epochs': 50},
    # (B) bigger network, a bit more dropout, larger batch
    {'layers': [512, 256, 128], 'dropout': 0.30, 'learning_rate': 1e-3, 'batch_size': 64, 'epochs': 60},
    # (C) smaller network, lower LR, more epochs
    {'layers': [128, 64, 32],  'dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 32, 'epochs': 80},
]

# 12. Define the neural network model architecture (parameterized)
def create_model(input_dim, params):
    model = Sequential([
        Dense(params['layers'][0], activation='relu', input_shape=(input_dim,)),
        Dropout(params['dropout']),
        Dense(params['layers'][1], activation='relu'),
        Dropout(params['dropout']),
        Dense(params['layers'][2], activation='relu'),
        Dropout(params['dropout']),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 13. Function to calculate specificity (unchanged)
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

# 14. Initialize storage for cross-validation run results (unchanged)
run_results = {
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'Balanced Accuracy': [],
    'AUROC': [],
    'F1 Score': []
}

# 15. Perform multiple runs (here, 5 runs) with 10-fold cross-validation in each run
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

        train_features_full = train_features_raw.copy()
        test_features       = test_features_raw.copy()
        train_features_full[:, num_start:num_end] = scaler_fold.transform(train_features_full[:, num_start:num_end])
        test_features[:,     num_start:num_end]   = scaler_fold.transform(test_features[:,     num_start:num_end])
        # ---------------------------------------------------------

        train_targets_full, test_targets = targets[train_index], targets[test_index]

        # ---- 90/10 validation split inside the training fold ----
        cut = int(len(train_features_full) * 0.9)
        X_tr, y_tr   = train_features_full[:cut], train_targets_full[:cut]
        X_val, y_val = train_features_full[cut:],  train_targets_full[cut:]

        # ---- Hyperparameter selection on validation set (by AUROC, fallback to accuracy) ----
        best_score, best_params = -np.inf, None
        for params in nn_param_list:
            m = create_model(X_tr.shape[1], params)
            m.fit(X_tr, y_tr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
            val_probs = m.predict(X_val, verbose=0).flatten()
            try:
                val_score = roc_auc_score(y_val, val_probs)
            except ValueError:
                # AUROC undefined if y_val has a single class; fallback to accuracy
                val_preds = (val_probs > 0.5).astype(int)
                val_score = accuracy_score(y_val, val_preds)
            if val_score > best_score:
                best_score, best_params = val_score, params

        # ---- Retrain best model on FULL fold training (train + val) ----
        model = create_model(train_features_full.shape[1], best_params)
        model.fit(train_features_full, train_targets_full,
                  epochs=best_params['epochs'],
                  batch_size=best_params['batch_size'],
                  verbose=0)

        # Predict on the test set (unchanged)
        test_probabilities = model.predict(test_features, verbose=0).flatten()
        predictions = (test_probabilities > 0.5).astype(int)

        # Calculate evaluation metrics (unchanged)
        accuracy = accuracy_score(test_targets, predictions)
        sensitivity = recall_score(test_targets, predictions)
        specificity = calculate_specificity(test_targets, predictions)
        balanced_accuracy = (sensitivity + specificity) / 2
        auroc = roc_auc_score(test_targets, test_probabilities)
        f1 = f1_score(test_targets, predictions)

        # Store metrics for this fold (unchanged)
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

    print(f"Run {run+1} - Average Metrics:")
    print(f"  Accuracy: {np.mean(accuracies):.3f}")
    print(f"  Sensitivity: {np.mean(sensitivities):.3f}")
    print(f"  Specificity: {np.mean(specificities):.3f}")
    print(f"  Balanced Accuracy: {np.mean(balanced_accuracies):.3f}")
    print(f"  AUROC: {np.mean(aurocs):.3f}")
    print(f"  F1 Score: {np.mean(f1_scores):.3f}")

# 16. Calculate and print the overall average and standard deviation of the metrics over all runs
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")
