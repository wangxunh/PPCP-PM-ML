# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:10:04 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# Function to generate ECFPs from SMILES
def smiles_to_ecfp(smiles, radius=2, n_bits=1024):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smiles)
    return np.array(gen.GetFingerprint(mol)) if mol else np.zeros((n_bits,))

# Read data
data = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')
data['ECFP'] = data['SMILES'].apply(smiles_to_ecfp)

# 1. Define numeric columns (ensure these match your CSV)
numeric_cols = [
        'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]

# 2. Define raw categorical columns to be one-hot encoded
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

# 3. One-hot encode the raw categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = ohe.fit_transform(data[categorical_cols])
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)

# 4. Combine numeric features and one-hot encoded categorical features
features_numeric = data[numeric_cols].values
other_features = np.hstack((features_numeric, cat_features))

# 5. Combine with ECFP features
ecfp_features = np.vstack(data['ECFP'].values)
features = np.hstack((ecfp_features, other_features))

# 6. NOTE: Do NOT standardize the full matrix here (avoids leakage).
#    Only numeric columns will be z-scored inside each CV fold.

# 7. Define target variable
y = data['Removal efficiency'].values

# 8. Setup KFold
kf = KFold(n_splits=10, shuffle=True)

# numeric indices inside `features`:
# [0 : ecfp_size) are ECFP bits
# [ecfp_size : ecfp_size+len(numeric_cols)) are numeric cols
ecfp_size = ecfp_features.shape[1]
num_start = ecfp_size
num_end = ecfp_size + len(numeric_cols)

# ---------------------------
# SVM: params list (vector)
# ---------------------------
svm_params1 = {'C': 1.0,  'epsilon': 0.10, 'kernel': 'rbf', 'gamma': 'scale'}
svm_params2 = {'C': 10.0, 'epsilon': 0.50, 'kernel': 'rbf', 'gamma': 'scale'}
svm_params3 = {'C': 1.0,  'epsilon': 0.05, 'kernel': 'rbf', 'gamma': 'scale'}
svm_params_list = [svm_params1, svm_params2, svm_params3]


# 10. Perform multiple runs
n_runs = 5
run_results = []

for run in range(n_runs):
    all_measured = []
    all_predicted = []

    # 10-fold CV (split on raw `features` to avoid leakage)
    for train_index, test_index in kf.split(features):
        best_score = -np.inf
        best_param = None

        # Raw train/test
        X_train_raw, X_test_raw = features[train_index], features[test_index]
        y_train,     y_test     = y[train_index], y[test_index]

        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric columns (no leakage)
        # ---------------------------------------------------------
        scaler_fold = StandardScaler()
        scaler_fold.fit(X_train_raw[:, num_start:num_end])  # fit on TRAIN numeric only

        train_features = X_train_raw.copy()
        test_features  = X_test_raw.copy()

        train_features[:, num_start:num_end] = scaler_fold.transform(
            X_train_raw[:, num_start:num_end]
        )
        test_features[:, num_start:num_end] = scaler_fold.transform(
            X_test_raw[:, num_start:num_end]
        )
        # ---------------------------------------------------------

        # 90% train / 10% val from the training portion
        cut = int(len(train_features) * 0.9)
        train_features_new = train_features[:cut]
        train_targets_new  = y_train[:cut]
        valid_features     = train_features[cut:]
        valid_targets      = y_train[cut:]

        # Pick best params on the validation set
        for svm_params in svm_params_list:
            model = SVR(**svm_params)
            model.fit(train_features_new, train_targets_new)
            val_score = model.score(valid_features, valid_targets)  # R^2 on val
            if val_score > best_score:
                best_score = val_score
                best_param = svm_params

        # Retrain with best params on the full training fold (train+val)
        model = SVR(**best_param)
        model.fit(train_features, y_train)
        print(best_param)

        # Predict on the test fold
        predictions = model.predict(test_features)
        predictions = np.clip(predictions, 0, 1)  # keep in [0,1] if needed

        all_measured.extend(y_test)
        all_predicted.extend(predictions)

    # Metrics for the run
    mse = mean_squared_error(all_measured, all_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_measured, all_predicted)
    r2  = r2_score(all_measured, all_predicted)
    run_results.append((r2, mae, rmse, mse))

    print(f"Run {run+1} - R2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}, MSE: {mse:.3f}")

    # Plot aggregated predictions
    plt.figure(figsize=(6, 4))
    plt.scatter(all_measured, all_predicted, c='blue', alpha=0.5, label='Predictions')
    lo, hi = min(all_measured), max(all_measured)
    plt.plot([lo, hi], [lo, hi], 'r-', label='Ideal')
    plt.xlabel('Measured Efficiency', fontsize=12)
    plt.ylabel('Predicted Efficiency', fontsize=12)
    plt.title(f'$R^2$ = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}, SVM-ECFP', fontsize=11)
    plt.xticks(fontsize=12); plt.yticks(fontsize=12);
    plt.show()

# 11. Calculate overall metrics
metrics = np.array(run_results)
means = metrics.mean(axis=0)
stds = metrics.std(axis=0)

print(f"Overall Metrics - R2 Mean: {means[0]:.3f}, Std Dev: {stds[0]:.3f}")
print(f"Overall Metrics - MAE Mean: {means[1]:.3f}, Std Dev: {stds[1]:.3f}")
print(f"Overall Metrics - RMSE Mean: {means[2]:.3f}, Std Dev: {stds[2]:.3f}")
print(f"Overall Metrics - MSE Mean: {means[3]:.3f}, Std Dev: {stds[3]:.3f}")
