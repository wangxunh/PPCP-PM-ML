# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 22:46:31 2025

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Read data
df = pd.read_csv('PPCP_PM_20251218.csv', encoding='utf-8-sig')
df = df.dropna(how="all").copy()

# 2) Columns
numeric_cols = [
    'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]
categorical_cols = [
    'Photocatalyst category','Membrane materials','Membrane type',
    'Light frequency','Hybrid methods'
]

# (optional) clean numeric artifacts such as NBSP / 'Â'
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
              .str.replace("\u00a0", "", regex=False)
              .str.replace("Â", "", regex=False)
              .str.strip(),
        errors="coerce"
    )

for c in numeric_cols:
    if df[c].dtype == "object":
        df[c] = clean_numeric(df[c])
    else:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# drop rows with missing values in the used columns
df = df.dropna(subset=numeric_cols + categorical_cols + ['Removal efficiency']).reset_index(drop=True)

# 3) One-hot encode categorical features (simple)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = ohe.fit_transform(df[categorical_cols])
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)

# 4) Stack numeric + encoded categorical features
features_numeric = df[numeric_cols].values
X = np.hstack((features_numeric, cat_features))
y = df['Removal efficiency'].values

feature_names = numeric_cols + list(encoded_feature_names)

# 5) K-Fold cross-validation setup
kf = KFold(n_splits=10, shuffle=True)

# ---------------------------
# Multiple runs
# ---------------------------
n_runs = 5
run_results = []

for run in range(n_runs):
    all_measured = []
    all_predicted = []

    # 10-fold CV
    for train_index, test_index in kf.split(X):
        # (keep the same "best_param" structure, but there is nothing to tune)
        best_param = {"model": "LinearRegression"}  # placeholder to keep prints consistent

        # ---- Split FIRST on raw X (avoid leakage) ----
        train_features, test_features = X[train_index].copy(), X[test_index].copy()
        train_targets,  test_targets  = y[train_index],       y[test_index]

        # ---- Standardize ONLY numeric columns, fit scaler on training fold ONLY ----
        scaler = StandardScaler()
        train_features[:, :len(numeric_cols)] = scaler.fit_transform(train_features[:, :len(numeric_cols)])
        test_features[:,  :len(numeric_cols)] = scaler.transform(test_features[:,  :len(numeric_cols)])

        # 90/10 split inside training fold (kept for consistency)
        cut = int(len(train_features) * 0.9)
        train_features_new = train_features[:cut]
        train_targets_new  = train_targets[:cut]
        valid_features     = train_features[cut:]
        valid_targets      = train_targets[cut:]

        model = LinearRegression()
        model.fit(train_features_new, train_targets_new)

        # validation R2 (printed like SVM logic, but not used for selection)
        val_pred = model.predict(valid_features)
        val_score = r2_score(valid_targets, val_pred)

        # Retrain on full training fold
        model = LinearRegression()
        model.fit(train_features, train_targets)

        print(best_param)

        predictions = model.predict(test_features)
        predictions = np.clip(predictions, 0, 1)  # keep in [0,1] if desired

        all_measured.extend(test_targets)
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
    plt.title(f'$R^2$ = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}, Linear Regression', fontsize=11)
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.show()

# ---------------------------
# Overall metrics
# ---------------------------
metrics = np.array(run_results)
means = metrics.mean(axis=0)
stds  = metrics.std(axis=0)

print(f"Overall Metrics - R^2 Mean: {means[0]:.3f}, Std Dev: {stds[0]:.3f}")
print(f"Overall Metrics - MAE Mean: {means[1]:.3f}, Std Dev: {stds[1]:.3f}")
print(f"Overall Metrics - RMSE Mean: {means[2]:.3f}, Std Dev: {stds[2]:.3f}")
print(f"Overall Metrics - MSE Mean: {means[3]:.3f}, Std Dev: {stds[3]:.3f}")
