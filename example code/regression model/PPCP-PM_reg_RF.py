# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:38:44 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
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

# 12. Candidate parameter combinations (only vary the shown parameters)
rf_param_list = [
    # Baseline (your original)
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
    # Slightly deeper and larger forest
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
    # Shallower but more regularized
    {"n_estimators": 200, "max_depth": 20, "min_samples_split": 4, "min_samples_leaf": 2},
    # Fewer estimators but deeper trees
    {"n_estimators": 50, "max_depth": 40, "min_samples_split": 2, "min_samples_leaf": 1},
    # Moderate depth with more conservative split
    {"n_estimators": 150, "max_depth": 25, "min_samples_split": 3, "min_samples_leaf": 2},
]

# 13. Perform multiple runs
n_runs = 5
run_results = []

for run in range(n_runs):
    all_measured, all_predicted = [], []

    print(f"\n=== Run {run+1} ===")

    # IMPORTANT: split on raw X (not scaled) to avoid leakage
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):

        # Internal 90/10 split for validation (kept as in your code)
        cut = int(len(tr_idx) * 0.9)
        inner_tr_idx, inner_val_idx = tr_idx[:cut], tr_idx[cut:]

        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric columns (no leakage)
        # ---------------------------------------------------------
        scaler_fold = StandardScaler()
        scaler_fold.fit(X[tr_idx][:, num_start:num_end])  # fit on TRAIN-FOLD numeric only

        # Transform needed subsets (train/val/test)
        X_tr_raw = X[inner_tr_idx].copy()
        X_val_raw = X[inner_val_idx].copy()
        X_test_raw = X[te_idx].copy()

        X_tr_raw[:, num_start:num_end] = scaler_fold.transform(X_tr_raw[:, num_start:num_end])
        X_val_raw[:, num_start:num_end] = scaler_fold.transform(X_val_raw[:, num_start:num_end])
        X_test_raw[:, num_start:num_end] = scaler_fold.transform(X_test_raw[:, num_start:num_end])

        X_tr, y_tr = X_tr_raw, y[inner_tr_idx]
        X_val, y_val = X_val_raw, y[inner_val_idx]
        X_test, y_test = X_test_raw, y[te_idx]
        # ---------------------------------------------------------

        # --- Model selection on validation set ---
        best_score = -np.inf
        best_params = None

        for p in rf_param_list:
            rf = RandomForestRegressor(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                min_samples_split=p["min_samples_split"],
                min_samples_leaf=p["min_samples_leaf"],
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_tr, y_tr)
            val_pred = rf.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            if val_r2 > best_score:
                best_score = val_r2
                best_params = p

        # Print the best parameters for this fold
        print(best_params)

        # --- Retrain best model on full training fold (train+val) ---
        X_fold_train_raw = X[tr_idx].copy()
        X_fold_train_raw[:, num_start:num_end] = scaler_fold.transform(
            X_fold_train_raw[:, num_start:num_end]
        )
        y_fold_train = y[tr_idx]

        rf_best = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        rf_best.fit(X_fold_train_raw, y_fold_train)

        preds = rf_best.predict(X_test)
        all_measured.extend(y_test)
        all_predicted.extend(preds)

    # 14. Metrics for this run
    mse = mean_squared_error(all_measured, all_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_measured, all_predicted)
    r2  = r2_score(all_measured, all_predicted)
    run_results.append((r2, mae, rmse, mse))

    print(f"\nRun {run+1} Summary - R²: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}, MSE: {mse:.3f}\n")

    # 15. Plot results
    plt.figure(figsize=(6, 4))
    plt.scatter(all_measured, all_predicted, c='blue', alpha=0.5)
    lo, hi = min(all_measured), max(all_measured)
    plt.plot([lo, hi], [lo, hi], 'r-')
    plt.xlabel('Measured Efficiency', fontsize=12)
    plt.ylabel('Predicted Efficiency', fontsize=12)
    plt.title(f'$R^2$ = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}, RF', fontsize=11)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# 17. Summarize metrics over runs
metrics = np.array(run_results)
means = metrics.mean(axis=0)
stds = metrics.std(axis=0)

print(f"Overall Metrics - R^2 Mean: {means[0]:.3f}, Std Dev: {stds[0]:.3f}")
print(f"Overall Metrics - MAE Mean: {means[1]:.3f}, Std Dev: {stds[1]:.3f}")
print(f"Overall Metrics - RMSE Mean: {means[2]:.3f}, Std Dev: {stds[2]:.3f}")
print(f"Overall Metrics - MSE Mean: {means[3]:.3f}, Std Dev: {stds[3]:.3f}")
