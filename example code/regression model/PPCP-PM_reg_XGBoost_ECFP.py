# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:04:54 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

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
#     Only numeric columns (features_numeric) will be z-scored inside each training fold.

# 7. Define target variable
y = data['Removal efficiency'].values

# 8. Setup KFold
kf = KFold(n_splits=10, shuffle=True)

# Indices of numeric block inside `features`
# features = [ECFP(0:1024), numeric(1024:1024+len(numeric_cols)), onehot(rest)]
ecfp_size = ecfp_features.shape[1]
n_num = len(numeric_cols)
num_start, num_end = ecfp_size, ecfp_size + n_num

# candidate hyperparameter sets
xgb_param_list = [
    # (A) original
    {
        'objective': 'reg:squarederror',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    # (B) deeper trees, slower learning, more estimators
    {
        'objective': 'reg:squarederror',
        'n_estimators': 400,
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_child_weight': 3,
        'subsample': 0.9,
        'colsample_bytree': 0.8
    },
    # (C) shallower trees, faster learning
    {
        'objective': 'reg:squarederror',
        'n_estimators': 300,
        'learning_rate': 0.2,
        'max_depth': 4,
        'min_child_weight': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.6
    },
]

# 14. Multiple runs
n_runs = 5
run_results = []

for run in range(n_runs):
    all_measured = []
    all_predicted = []
    print(f"\n=== Run {run+1} ===")

    # IMPORTANT: split on raw `features` to avoid leakage
    for train_index, test_index in kf.split(features):
        # fold train/test (raw)
        train_features_full_raw, test_features_raw = features[train_index].copy(), features[test_index].copy()
        train_targets_full,      test_targets     = y[train_index], y[test_index]

        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric columns (no leakage)
        # ---------------------------------------------------------
        scaler_fold = StandardScaler()
        scaler_fold.fit(train_features_full_raw[:, num_start:num_end])  # fit on TRAIN numeric only

        train_features_full = train_features_full_raw.copy()
        test_features       = test_features_raw.copy()

        train_features_full[:, num_start:num_end] = scaler_fold.transform(train_features_full[:, num_start:num_end])
        test_features[:, num_start:num_end]       = scaler_fold.transform(test_features[:, num_start:num_end])
        # ---------------------------------------------------------

        # 90/10 validation split inside the training fold
        cut = int(len(train_features_full) * 0.9)
        X_tr, y_tr = train_features_full[:cut], train_targets_full[:cut]
        X_val, y_val = train_features_full[cut:], train_targets_full[cut:]

        # choose best params by validation R^2
        best_score = -np.inf
        best_params = None
        for p in xgb_param_list:
            model = xgb.XGBRegressor(**p)
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            if val_r2 > best_score:
                best_score = val_r2
                best_params = p

        # Print the best parameters for this fold
        print(best_params)

        # retrain best model on full training (train + val) and evaluate on test fold
        model = xgb.XGBRegressor(**best_params)
        model.fit(train_features_full, train_targets_full)
        predictions = model.predict(test_features)

        all_measured.extend(test_targets)
        all_predicted.extend(predictions)

    mse = mean_squared_error(all_measured, all_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_measured, all_predicted)
    r2 = r2_score(all_measured, all_predicted)
    run_results.append((r2, mae, rmse, mse))

    print(f"Run {run+1} - R^2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}, MSE: {mse:.3f}")

    plt.figure(figsize=(6, 4))
    plt.scatter(all_measured, all_predicted, c='blue', alpha=0.5)
    plt.plot([min(all_measured), max(all_measured)], [min(all_measured), max(all_measured)], 'r-')
    plt.xlabel('Measured Efficiency', fontsize=12)
    plt.ylabel('Predicted Efficiency', fontsize=12)
    plt.title(f'$R^2$ = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}, XGB-ECFP', fontsize=11)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

metrics = np.array(run_results)
means = metrics.mean(axis=0)
stds = metrics.std(axis=0)

print(f"Overall Metrics - R^2 Mean: {means[0]:.3f}, Std Dev: {stds[0]:.3f}")
print(f"Overall Metrics - MAE Mean: {means[1]:.3f}, Std Dev: {stds[1]:.3f}")
print(f"Overall Metrics - RMSE Mean: {means[2]:.3f}, Std Dev: {stds[2]:.3f}")
print(f"Overall Metrics - MSE Mean: {means[3]:.3f}, Std Dev: {stds[3]:.3f}")
