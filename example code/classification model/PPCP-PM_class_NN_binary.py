# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:23:02 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

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

# 8. Select features (X) and define binary classification target (y)
X = df_final[final_feature_list].values
y = (df_final['Removal efficiency'] > 0.5).astype(int).values

# -------------------------------------------------------------------
# IMPORTANT: Do NOT z-score ALL features here (avoids leakage).
# Only z-score numeric base columns inside each training fold.
# -------------------------------------------------------------------
num_start, num_end = 0, len(base_features)

# 10. K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# --- Hyperparameter sets (include your original as Option A) ---
nn_param_list = [
    # (A) original settings
    {'layers': [256, 128, 64], 'dropout': 0.20, 'learning_rate': 1e-3, 'batch_size': 32, 'epochs': 50},
    # (B) larger network, a bit more dropout, larger batch
    {'layers': [512, 256, 128], 'dropout': 0.30, 'learning_rate': 1e-3, 'batch_size': 64, 'epochs': 60},
    # (C) smaller network, slightly lower LR, more epochs
    {'layers': [128, 64, 32],  'dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 32, 'epochs': 80},
]

# 11. Define the neural network model (now parameterized; keeps same architecture style)
def create_model(input_dim, params):
    model = Sequential([
        Dense(params['layers'][0], activation='relu', input_shape=(input_dim,)),
        Dropout(params['dropout']),
        Dense(params['layers'][1], activation='relu'),
        Dropout(params['dropout']),
        Dense(params['layers'][2], activation='relu'),
        Dropout(params['dropout']),
        Dense(1, activation='sigmoid')  # binary classification output
    ])
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 12. Initialize storage for results of each run (unchanged)
run_results = {
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'Balanced Accuracy': [],
    'AUROC': [],
    'F1 Score': []
}

n_runs = 3
for run in range(n_runs):
    accuracies = []
    sensitivities = []
    specificities = []
    balanced_accuracies = []
    aurocs = []
    f1_scores = []

    # IMPORTANT: split on raw X (not scaled) to avoid leakage
    for train_index, test_index in kf.split(X):

        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric columns (no leakage)
        # ---------------------------------------------------------
        train_features_raw = X[train_index].copy()
        test_features_raw  = X[test_index].copy()

        scaler_fold = StandardScaler()
        scaler_fold.fit(train_features_raw[:, num_start:num_end])  # fit on TRAIN numeric only

        train_features_full = train_features_raw.copy()
        test_features       = test_features_raw.copy()
        train_features_full[:, num_start:num_end] = scaler_fold.transform(train_features_full[:, num_start:num_end])
        test_features[:,     num_start:num_end]   = scaler_fold.transform(test_features[:,     num_start:num_end])
        # ---------------------------------------------------------

        train_targets_full, test_targets = y[train_index], y[test_index]

        # ---- 90/10 validation split inside the training fold ----
        cut = int(len(train_features_full) * 0.9)
        X_tr, y_tr   = train_features_full[:cut], train_targets_full[:cut]
        X_val, y_val = train_features_full[cut:], train_targets_full[cut:]

        # ---- Hyperparameter selection on validation set (by AUROC; fallback to accuracy if needed) ----
        best_score, best_params = -np.inf, None
        for params in nn_param_list:
            m = create_model(X_tr.shape[1], params)
            m.fit(X_tr, y_tr, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
            val_probs = m.predict(X_val, verbose=0).flatten()
            try:
                val_score = roc_auc_score(y_val, val_probs)
            except ValueError:
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

        # ---- Predict on the held-out test fold ----
        test_probabilities = model.predict(test_features, verbose=0).flatten()
        predictions = (test_probabilities > 0.5).astype(int)

        # ---- Calculate metrics (unchanged outputs) ----
        accuracy = accuracy_score(test_targets, predictions)
        sensitivity = recall_score(test_targets, predictions)
        specificity = calculate_specificity(test_targets, predictions)
        balanced_accuracy = (sensitivity + specificity) / 2
        auroc = roc_auc_score(test_targets, test_probabilities)
        f1 = f1_score(test_targets, predictions)

        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        balanced_accuracies.append(balanced_accuracy)
        aurocs.append(auroc)
        f1_scores.append(f1)

    # Aggregate per-run (unchanged printing)
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

# 13. Calculate and print overall average and std of metrics over runs (unchanged)
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")
