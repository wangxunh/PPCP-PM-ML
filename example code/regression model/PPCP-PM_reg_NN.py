# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:02:56 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


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

# ---------------------------
# NN: params list (vector)
# ---------------------------
nn_params1 = {
    'layers': [512, 256, 128],
    'dropout': 0.10,
    'activation': 'relu',
    'optimizer_lr': 1e-3,     # <-- store LR, not an optimizer object
    'batch_size': 32,
    'epochs': 50,
    'output_activation': 'sigmoid'
}
nn_params2 = {
    'layers': [256, 128, 64],
    'dropout': 0.20,
    'activation': 'relu',
    'optimizer_lr': 1e-3,
    'batch_size': 32,
    'epochs': 50,
    'output_activation': 'sigmoid'
}
nn_params3 = {
    'layers': [512, 256],
    'dropout': 0.10,
    'activation': 'tanh',
    'optimizer_lr': 1e-3,
    'batch_size': 32,
    'epochs': 50,
    'output_activation': 'sigmoid'
}
nn_params_list = [nn_params1, nn_params2, nn_params3]

def build_nn(input_dim, params):
    model = Sequential([Input(shape=(input_dim,))])
    for units in params['layers']:
        model.add(Dense(units, activation=params['activation']))
        model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation=params['output_activation']))
    # fresh optimizer instance for every build
    opt = Adam(learning_rate=params['optimizer_lr'])
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

# ---------------------------
# Multiple runs
# ---------------------------
n_runs = 5
run_results = []

# numeric indices = the first len(base_features) columns in final_feature_list
n_num = len(base_features)
num_start, num_end = 0, n_num

for run in range(n_runs):
    all_measured = []
    all_predicted = []

    # 10-fold CV (split on raw X to avoid leakage)
    for train_index, test_index in kf.split(X):
        best_score = -np.inf
        best_param = None

        # Raw fold split
        train_features_full_raw, test_features_raw = X[train_index], X[test_index]
        train_targets_full,      test_targets      = y[train_index], y[test_index]

        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric columns (no leakage)
        # ---------------------------------------------------------
        scaler_fold = StandardScaler()
        scaler_fold.fit(train_features_full_raw[:, num_start:num_end])  # fit on TRAIN numeric only

        train_features_full = train_features_full_raw.copy()
        test_features       = test_features_raw.copy()

        train_features_full[:, num_start:num_end] = scaler_fold.transform(
            train_features_full_raw[:, num_start:num_end]
        )
        test_features[:, num_start:num_end] = scaler_fold.transform(
            test_features_raw[:, num_start:num_end]
        )
        # ---------------------------------------------------------

        cut = int(len(train_features_full) * 0.9)  # 90% train, 10% val
        train_features = train_features_full[:cut]
        train_targets  = train_targets_full[:cut]
        val_features   = train_features_full[cut:]
        val_targets    = train_targets_full[cut:]

        # Parameter selection on validation set (by R²)
        for nn_params in nn_params_list:
            model = build_nn(train_features_full.shape[1], nn_params)
            model.fit(
                train_features, train_targets,
                validation_data=(val_features, val_targets),
                epochs=nn_params['epochs'],
                batch_size=nn_params['batch_size'],
                verbose=0
            )
            val_pred = model.predict(val_features, verbose=0).flatten()
            # Clip if target is strictly within [0,1]
            val_pred = np.clip(val_pred, 0, 1)
            val_r2 = r2_score(val_targets, val_pred)
            if val_r2 > best_score:
                best_score = val_r2
                best_param = nn_params

        # Retrain best configuration on the full training fold
        model = build_nn(train_features_full.shape[1], best_param)
        print(best_param)
        model.fit(
            train_features_full, train_targets_full,
            validation_data=(val_features, val_targets),
            epochs=best_param['epochs'],
            batch_size=best_param['batch_size'],
            verbose=0
        )

        # Predict on the held-out test fold
        predictions = model.predict(test_features, verbose=0).flatten()
        predictions = np.clip(predictions, 0, 1)  # keep in [0,1] if needed

        all_measured.extend(test_targets)
        all_predicted.extend(predictions)

    # Metrics for the run
    mse = mean_squared_error(all_measured, all_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_measured, all_predicted)
    r2  = r2_score(all_measured, all_predicted)
    run_results.append((r2, mae, rmse, mse))

    print(f"Run {run+1} - R^2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}, MSE: {mse:.3f}")

    # Plot aggregated predictions (no legend/grid)
    plt.figure(figsize=(6, 4))
    plt.scatter(all_measured, all_predicted, c='blue', alpha=0.5)
    lo, hi = min(all_measured), max(all_measured)
    plt.plot([lo, hi], [lo, hi], 'r-')
    plt.xlabel('Measured Efficiency', fontsize=12)
    plt.ylabel('Predicted Efficiency', fontsize=12)
    plt.title(f'$R^2$ = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}, NN', fontsize=11)
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.show()
    
metrics = np.array(run_results)
means = metrics.mean(axis=0)
stds = metrics.std(axis=0)

print(f"Overall Metrics - R^2 Mean: {means[0]:.3f}, Std Dev: {stds[0]:.3f}")
print(f"Overall Metrics - MAE Mean: {means[1]:.3f}, Std Dev: {stds[1]:.3f}")
print(f"Overall Metrics - RMSE Mean: {means[2]:.3f}, Std Dev: {stds[2]:.3f}")
print(f"Overall Metrics - MSE Mean: {means[3]:.3f}, Std Dev: {stds[3]:.3f}")
