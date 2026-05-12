# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:12:10 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Read data
df = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')

# 2. Define which columns are categorical and need one-hot encoding
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

# 3–4. Initialize OneHotEncoder and transform categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_array = ohe.fit_transform(df[categorical_cols])
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names)

# 5. Combine numeric columns and one-hot encoded columns
df_numerical = df.drop(columns=categorical_cols)
df_final = pd.concat([df_numerical, df_encoded], axis=1)

# 6. Define numeric base columns
base_features = [
        'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]

# 7–8. Get final feature list (numeric + encoded)
one_hot_cols = list(encoded_feature_names)
final_feature_list = base_features + one_hot_cols

# 9. Select features (X) and target (y)
X = df_final[final_feature_list].values
y = df_final['Removal efficiency'].values

# 10. NOTE: Do NOT standardize the full matrix here (avoids leakage).
#     Only numeric features (base_features) will be z-scored inside each training fold.

# 11. K-Fold cross-validation setup
kf = KFold(n_splits=10, shuffle=True)

# numeric indices = the first len(base_features) columns in final_feature_list
n_num = len(base_features)
num_start, num_end = 0, n_num

# 3 candidate hyperparameter sets
xgb_param_list = [
    # (A) your original
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
    # (C) shallower trees, faster learning, stronger column subsampling
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

# Perform multiple runs
n_runs = 5
run_results = []

for run in range(n_runs):
    all_measured = []
    all_predicted = []
    print(f"\n=== Run {run+1} ===")

    # Cross-validation loop (IMPORTANT: split on raw X to avoid leakage)
    for fold_id, (train_index, test_index) in enumerate(kf.split(X), start=1):

        # Raw train/test for this fold
        X_train_raw, X_test_raw = X[train_index].copy(), X[test_index].copy()
        y_train_full, y_test = y[train_index], y[test_index]

        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric columns (no leakage)
        # ---------------------------------------------------------
        scaler_fold = StandardScaler()
        scaler_fold.fit(X_train_raw[:, num_start:num_end])  # fit on TRAIN numeric only
        X_train_raw[:, num_start:num_end] = scaler_fold.transform(X_train_raw[:, num_start:num_end])
        X_test_raw[:, num_start:num_end]  = scaler_fold.transform(X_test_raw[:, num_start:num_end])
        # ---------------------------------------------------------

        # Create a validation split inside the training fold (90/10)
        cut = int(len(X_train_raw) * 0.9)
        X_tr, y_tr = X_train_raw[:cut], y_train_full[:cut]
        X_val, y_val = X_train_raw[cut:], y_train_full[cut:]

        # ---- Select best params on the validation set (by R^2) ----
        best_score = -np.inf
        best_params = None

        for p in xgb_param_list:
            model = xgb.XGBRegressor(**p, n_jobs=-1)
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            if val_r2 > best_score:
                best_score = val_r2
                best_params = p

        # (optional) print the best params chosen for this fold
        print(best_params)

        # ---- Retrain best model on full fold training (train + val) ----
        model = xgb.XGBRegressor(**best_params, n_jobs=-1)
        model.fit(X_train_raw, y_train_full)

        # Predict on the held-out test fold
        preds = model.predict(X_test_raw)

        # Collect predictions
        all_measured.extend(y_test)
        all_predicted.extend(preds)

    # Calculate metrics for the run
    mse = mean_squared_error(all_measured, all_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_measured, all_predicted)
    r2 = r2_score(all_measured, all_predicted)
    run_results.append((r2, mae, rmse, mse))

    print(f"Run {run+1} - R^2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}, MSE: {mse:.3f}")

    # Plot results
    plt.figure(figsize=(6, 4))
    plt.scatter(all_measured, all_predicted, c='blue', alpha=0.5)
    lo, hi = min(all_measured), max(all_measured)
    plt.plot([lo, hi], [lo, hi], 'r-')
    plt.xlabel('Measured Efficiency', fontsize=12)
    plt.ylabel('Predicted Efficiency', fontsize=12)
    plt.title(f'$R^2$ = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}, XGB', fontsize=11)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# Calculate overall metrics
metrics = np.array(run_results)
means = metrics.mean(axis=0)
stds = metrics.std(axis=0)

# Print the overall average and standard deviation of the metrics over all runs
print(f"Overall Metrics - R^2 Mean: {means[0]:.3f}, Std Dev: {stds[0]:.3f}")
print(f"Overall Metrics - MAE Mean: {means[1]:.3f}, Std Dev: {stds[1]:.3f}")
print(f"Overall Metrics - RMSE Mean: {means[2]:.3f}, Std Dev: {stds[2]:.3f}")
print(f"Overall Metrics - MSE Mean: {means[3]:.3f}, Std Dev: {stds[3]:.3f}")
