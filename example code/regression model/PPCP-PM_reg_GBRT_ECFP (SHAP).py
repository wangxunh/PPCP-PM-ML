# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:21:07 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from IPython.display import SVG
import shap

import matplotlib.pyplot as plt
from io import BytesIO
import sys
from PIL import Image

# ---------------------------
# 1. ECFP generation
# ---------------------------
def smiles_to_ecfp(smiles, radius=2, n_bits=1024):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smiles)
    return np.array(gen.GetFingerprint(mol)) if mol is not None else np.zeros((n_bits,))

def smiles_to_ecfp_with_bitinfo(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        bitInfo = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bitInfo)
        return np.array(fp), bitInfo
    else:
        return np.zeros((n_bits,)), {}
    
def visualize_substructure(mol, bitInfo, bit):
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
    atom_indices = [ai for ai, radius in bitInfo.get(bit, [])]
    drawer.DrawMolecule(mol, highlightAtoms=atom_indices)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')
    return SVG(svg)

# ---------------------------
# 2. Read data
# ---------------------------
data = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')
data = data.dropna(how="all").copy()

# Generate ECFP features
data['ECFP'] = data['SMILES'].apply(smiles_to_ecfp)

# ---------------------------
# 3. Define columns for numeric & raw categorical
# ---------------------------
numeric_cols = [
        'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]

# Raw categorical columns (NOT pre-encoded)
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

# ---------------------------
# 4. One-hot encode the raw categorical columns
# ---------------------------
ohe = OneHotEncoder()
cat_features = ohe.fit_transform(data[categorical_cols])
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)

# Combine numeric features and one-hot encoded features
features_numeric = data[numeric_cols].values
other_features = np.hstack((features_numeric, cat_features.toarray()))

# Combine with ECFP features
ecfp_features = np.array(data['ECFP'].tolist())
features = np.hstack((ecfp_features, other_features))

# Define target
targets = data['Removal efficiency'].values

# ---------------------------
# 5. NOTE: Do NOT scale on the full dataset here (avoids leakage).
#    Also, do NOT z-score categorical/ECFP bits; only numeric features will be scaled
#    inside each training fold.
# ---------------------------

# ---------------------------
# 6. K-Fold cross-validation
# ---------------------------
kf = KFold(n_splits=10, shuffle=True)

# Gradient Boosting parameters
gb_params1 = {
        'n_estimators': 400,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'subsample': 0.8
}

gb_params2 = {
        'n_estimators': 400,
        'learning_rate': 0.001,
        'max_depth': 9,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 0.8
}

gb_params3 = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 0.8
}

gb_params_list = [gb_params1, gb_params2, gb_params3]


def bootstrap_ci_metrics(y_true, y_pred, B=5000, alpha=0.05):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    r2_vals = np.empty(B)
    mae_vals = np.empty(B)
    rmse_vals = np.empty(B)

    for _ in range(B):
        idx = np.random.randint(0, n, size=n)  # resample pairs
        yt = y_true[idx]
        yp = y_pred[idx]
        r2_vals[_]  = r2_score(yt, yp)
        mae_vals[_] = mean_absolute_error(yt, yp)
        rmse_vals[_]= np.sqrt(mean_squared_error(yt, yp))

    def pct_ci(arr):
        lo = np.percentile(arr, 100*(alpha/2))
        hi = np.percentile(arr, 100*(1-alpha/2))
        return lo, hi

    return {
        "R2":  (np.mean(r2_vals),)  + pct_ci(r2_vals),
        "MAE": (np.mean(mae_vals),) + pct_ci(mae_vals),
        "RMSE":(np.mean(rmse_vals),)+ pct_ci(rmse_vals),
    }

pooled_measured = []
pooled_predicted = []

# ---------------------------
# 7. Multiple runs
# ---------------------------
n_runs = 5
run_results = []
best_param = 0

# Indices for assembling fold features with numeric-only scaling
ecfp_size = ecfp_features.shape[1]
n_num = len(numeric_cols)
# numeric block is directly after ecfp block in `features`
num_start = ecfp_size
num_end = ecfp_size + n_num

for run in range(n_runs):
    all_measured = []
    all_predicted = []

    # Perform K-Fold cross-validation (split on raw, unscaled features to avoid leakage)
    for train_index, test_index in kf.split(features):
        
        best_score = -np.inf
        best_param = None
        
        train_features_raw, test_features_raw = features[train_index], features[test_index]
        train_targets, test_targets = targets[train_index], targets[test_index]

        # ---------------------------------------------------------
        # Fold-safe scaling: z-score ONLY numeric columns (no leakage)
        # ---------------------------------------------------------
        scaler_fold = StandardScaler()
        scaler_fold.fit(train_features_raw[:, num_start:num_end])  # fit on TRAIN numeric only

        train_features = train_features_raw.copy()
        test_features = test_features_raw.copy()

        train_features[:, num_start:num_end] = scaler_fold.transform(train_features_raw[:, num_start:num_end])
        test_features[:,  num_start:num_end] = scaler_fold.transform(test_features_raw[:,  num_start:num_end])
        # ---------------------------------------------------------

        train_features_new = train_features[:int(len(train_features) * 0.9)]
        train_targets_new = train_targets[:int(len(train_targets) * 0.9)]

        valid_feature = train_features[int(len(train_features) * 0.9):]
        valid_target = train_targets[int(len(train_features) * 0.9):]
        
        for gb_params in gb_params_list:
            # Initialize and train Gradient Boosting model
            model = GradientBoostingRegressor(**gb_params)
            model.fit(train_features_new, train_targets_new)
            val_score = model.score(valid_feature, valid_target)
            if val_score > best_score:
                best_param = gb_params
                best_score = val_score
        
        model = GradientBoostingRegressor(**best_param)
        model.fit(train_features, train_targets)
        print(best_param)

        # Predicting the efficiency
        predictions = model.predict(test_features)
        all_measured.extend(test_targets)
        all_predicted.extend(predictions)

    # Calculate metrics for the run
    mse = mean_squared_error(all_measured, all_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_measured, all_predicted)
    r2 = r2_score(all_measured, all_predicted)
    run_results.append((r2, mae, rmse, mse))

    # Print average metrics for this run
    print(f"Run {run+1} - R2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}, MSE: {mse:.3f}")
    
    # Plot aggregated predictions for the run
    plt.figure(figsize=(6, 4))
    plt.scatter(all_measured, all_predicted, c='blue', alpha=0.5, label='Predictions')
    plt.plot([min(all_measured), max(all_measured)], [min(all_measured), max(all_measured)], 'r-', label='Ideal')
    plt.xlabel('Measured Efficiency', fontsize=12)
    plt.ylabel('Predicted Efficiency', fontsize=12)
    plt.title(f'$R^2$ = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}, GBRT-ECFP', fontsize=11)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    pooled_measured.extend(all_measured)
    pooled_predicted.extend(all_predicted)


# ---------------------------
# 8. Overall metrics
# ---------------------------
metrics = np.array(run_results)
means = metrics.mean(axis=0)
stds = metrics.std(axis=0)

print(f"Overall Metrics - R2 Mean: {means[0]:.3f}, Std Dev: {stds[0]:.3f}")
print(f"Overall Metrics - MAE Mean: {means[1]:.3f}, Std Dev: {stds[1]:.3f}")
print(f"Overall Metrics - RMSE Mean: {means[2]:.3f}, Std Dev: {stds[2]:.3f}")
print(f"Overall Metrics - MSE Mean: {means[3]:.3f}, Std Dev: {stds[3]:.3f}")

ci = bootstrap_ci_metrics(pooled_measured, pooled_predicted, B=5000, alpha=0.05)
print("Overall Bootstrap 95% CI (pooled out-of-fold predictions across all runs):")
print(f"R2  mean={ci['R2'][0]:.3f}, 95% CI=({ci['R2'][1]:.3f}, {ci['R2'][2]:.3f})")
print(f"MAE mean={ci['MAE'][0]:.3f}, 95% CI=({ci['MAE'][1]:.3f}, {ci['MAE'][2]:.3f})")
print(f"RMSE mean={ci['RMSE'][0]:.3f}, 95% CI=({ci['RMSE'][1]:.3f}, {ci['RMSE'][2]:.3f})")

# ------------------------------------------------
# 9. SHAP or other analysis can follow
# ------------------------------------------------

# Build features_scaled for SHAP using numeric-only scaling fitted on full numeric block
# (This is for interpretation on the final model; CV leakage is already avoided above.)
scaler_full = StandardScaler()
features_scaled = features.copy()
features_scaled[:, num_start:num_end] = scaler_full.fit_transform(features[:, num_start:num_end])

# Retrain the model on the entire dataset
model = GradientBoostingRegressor(**best_param)
model.fit(features_scaled, targets)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features_scaled)

# 1) ECFP bits
ecfp_feature_names = [f"Bit_{i}" for i in range(ecfp_size)]

# 2) numeric_cols is your list of numeric columns
# 3) encoded_feature_names from your OneHotEncoder
feature_names = ecfp_feature_names + numeric_cols + list(encoded_feature_names)

subfeature_map = {}
for raw_col in categorical_cols:
    for enc_col in encoded_feature_names:
        if enc_col.startswith(raw_col + "_"):
            subfeature_map[enc_col] = raw_col
        elif enc_col == raw_col:
            subfeature_map[enc_col] = raw_col

# A2. unify_feature_name: returns raw col name if sub-feature
def unify_feature_name(name):
    return subfeature_map[name] if name in subfeature_map else name

# A3. For each column in feature_names, unify if it's a sub-feature
unified_feature_names = [unify_feature_name(fn) for fn in feature_names]

# A4. Sum absolute SHAP values by raw column
abs_shap_values = np.abs(shap_values)
M = len(feature_names)
raw_feature_aggregation = {}

for i in range(M):
    disp_name = unified_feature_names[i]
    # Mean absolute shap for sub-feature i
    col_mean = abs_shap_values[:, i].mean()
    # Accumulate that into the raw feature's total
    if disp_name not in raw_feature_aggregation:
        raw_feature_aggregation[disp_name] = 0.0
    raw_feature_aggregation[disp_name] += col_mean

# A5. Build a DataFrame & plot top 20
shap_df_agg = pd.DataFrame({
    'Feature': list(raw_feature_aggregation.keys()),
    'Mean Absolute SHAP Value': list(raw_feature_aggregation.values())
}).sort_values('Mean Absolute SHAP Value', ascending=False)

plt.figure(figsize=(10, 8))
top_k = 10
plt.barh(shap_df_agg['Feature'].head(top_k), shap_df_agg['Mean Absolute SHAP Value'].head(top_k))
plt.xlabel('Mean Absolute SHAP Value', fontsize=16)
plt.title('Top 10 Most Important Features - GBRT-ECFP Regression', fontsize=16)
plt.gca().invert_yaxis()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# C. Feature Importance of ECFP bits specifically
# ---------------------------
ecfp_shap_df = shap_df_agg[shap_df_agg['Feature'].str.startswith('Bit_')].copy()
ecfp_shap_df_sorted = ecfp_shap_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(ecfp_shap_df_sorted['Feature'].head(5),
        ecfp_shap_df_sorted['Mean Absolute SHAP Value'].head(5),
        orientation='vertical')

# Remove x-axis label and ticks
plt.xlabel('')         # Removes the label on the x-axis
plt.xticks([])         # Removes the tick labels on the x-axis

plt.ylabel('Mean Absolute SHAP Value', fontsize=16)
# plt.title('Top 5 ECFP Substructure - GBRT Regression', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


# ------------------------------------------------
# 10B. Visualize a specific ECFP bit in a molecule
# ------------------------------------------------
data['ECFP_with_bitinfo'] = data['SMILES'].apply(smiles_to_ecfp_with_bitinfo)

bit_of_interest = 175  # Example bit index
print("Bit of interest:", bit_of_interest)

# Find the first molecule where the bit is active
for idx, row in data.iterrows():
    _, bitInfo = row['ECFP_with_bitinfo']
    if bit_of_interest in bitInfo:
        mol = Chem.MolFromSmiles(row['SMILES'])
        display(visualize_substructure(mol, bitInfo, bit_of_interest))
        break
