# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:43:40 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, auc, classification_report
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

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
# 1. ECFP Generation
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
# 2. Read Data
# ---------------------------
data = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')
data['ECFP'] = data['SMILES'].apply(lambda x: smiles_to_ecfp(x))

# ---------------------------
# 3. Define numeric vs. raw categorical columns
# ---------------------------
numeric_cols = [
    'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]

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
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = ohe.fit_transform(data[categorical_cols])
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)

# Combine numeric features with one-hot encoded features
features_numeric = data[numeric_cols].values
other_features = np.hstack((features_numeric, cat_features))

# ---------------------------
# 5. Combine ECFP + other features
# ---------------------------
ecfp_features = np.array(data['ECFP'].tolist())
features = np.hstack((ecfp_features, other_features))

# ---------------------------
# 6. Binary classification target
# ---------------------------
targets = (data['Removal efficiency'] > 0.5).astype(int).values

# ---------------------------
# 7. IMPORTANT: Do NOT standardize full features here (avoids leakage).
#    Only z-score the numeric columns inside each training fold.
# ---------------------------
ecfp_size = ecfp_features.shape[1]
num_start = ecfp_size
num_end = ecfp_size + len(numeric_cols)

# ---------------------------
# 8. K-Fold cross-validation
# ---------------------------
kf = KFold(n_splits=10, shuffle=True)

# ---------------------------
# 9. Helper function to calculate specificity
# ---------------------------
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

# ---------------------------
# 10. Perform multiple runs
# ---------------------------
run_results = {
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'Balanced Accuracy': [],
    'AUROC': [],
    'F1 Score': []
}

gbr_param_list = [
    {'n_estimators': 100, 'learning_rate': 0.1,  'max_depth': 3},
    {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 3},
    {'n_estimators': 150, 'learning_rate': 0.20, 'max_depth': 4},
]

n_runs = 5
best_params_last = None  # keep last chosen params to reuse for full-data fit at end

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

        train_targets, test_targets = targets[train_index], targets[test_index]

        # ---- 90/10 validation split inside the training fold ----
        cut = int(len(train_features) * 0.9)
        X_tr, y_tr   = train_features[:cut], train_targets[:cut]
        X_val, y_val = train_features[cut:], train_targets[cut:]

        # ---- Hyperparameter selection on validation set (prefer AUROC; fallback to accuracy) ----
        best_score, best_params = -np.inf, None
        for p in gbr_param_list:
            clf_val = GradientBoostingClassifier(**p)
            clf_val.fit(X_tr, y_tr)
            try:
                val_probs = clf_val.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, val_probs)
            except ValueError:
                score = accuracy_score(y_val, clf_val.predict(X_val))
            if score > best_score:
                best_score, best_params = score, p

        best_params_last = best_params

        # ---- Retrain best model on FULL fold training (train + val) ----
        classifier = GradientBoostingClassifier(**best_params)
        classifier.fit(train_features, train_targets)

        # Predict on the test set
        predictions = classifier.predict(test_features)
        test_probabilities = classifier.predict_proba(test_features)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(test_targets, predictions)
        sensitivity = recall_score(test_targets, predictions)
        specificity = calculate_specificity(test_targets, predictions)
        balanced_accuracy = (sensitivity + specificity) / 2
        auroc = roc_auc_score(test_targets, test_probabilities)
        f1 = f1_score(test_targets, predictions)

        # Store metrics for this fold
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        balanced_accuracies.append(balanced_accuracy)
        aurocs.append(auroc)
        f1_scores.append(f1)

    # Store average metrics for this run
    run_results['Accuracy'].append(np.mean(accuracies))
    run_results['Sensitivity'].append(np.mean(sensitivities))
    run_results['Specificity'].append(np.mean(specificities))
    run_results['Balanced Accuracy'].append(np.mean(balanced_accuracies))
    run_results['AUROC'].append(np.mean(aurocs))
    run_results['F1 Score'].append(np.mean(f1_scores))

    # Print average metrics for this run
    print(f"Run {run+1} - Average Metrics:")
    print(f"  Accuracy: {np.mean(accuracies):.3f}")
    print(f"  Sensitivity: {np.mean(sensitivities):.3f}")
    print(f"  Specificity: {np.mean(specificities):.3f}")
    print(f"  Balanced Accuracy: {np.mean(balanced_accuracies):.3f}")
    print(f"  AUROC: {np.mean(aurocs):.4f}")
    print(f"  F1 Score: {np.mean(f1_scores):.3f}")

# ---------------------------
# 11. Print overall average and std
# ---------------------------
for metric in run_results:
    values = run_results[metric]
    print(f"{metric} - Mean: {np.mean(values):.3f}, Std Dev: {np.std(values):.3f}")

# ------------------------------------------------
# Feature Importance Section
# ------------------------------------------------

# For interpretation plots on full data (not CV evaluation), scale ONLY numeric columns on full dataset
scaler_full = StandardScaler()
scaler_full.fit(features[:, num_start:num_end])

features_scaled_full = features.copy()
features_scaled_full[:, num_start:num_end] = scaler_full.transform(features_scaled_full[:, num_start:num_end])

# Use last selected hyperparameters (minimal change to your structure)
if best_params_last is None:
    best_params_last = gbr_param_list[0]

classifier = GradientBoostingClassifier(**best_params_last)
classifier.fit(features_scaled_full, targets)

# Initialize the SHAP Explainer
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(features_scaled_full)

# 1) ECFP bits: label them as Bit_0, Bit_1, ...
ecfp_feature_names = [f"Bit_{i}" for i in range(ecfp_size)]

# 2) Numeric + encoded columns
other_feature_names = list(numeric_cols) + list(encoded_feature_names)

# Final feature names = ECFP bits + numeric/encoded columns
feature_names = ecfp_feature_names + other_feature_names

# -----------------------------
# A. Top-20 Chart (Aggregated by Raw Column)
# -----------------------------
subfeature_map = {}
for raw_col in categorical_cols:
    for enc_col in encoded_feature_names:
        if enc_col.startswith(raw_col + "_"):
            subfeature_map[enc_col] = raw_col
        elif enc_col == raw_col:
            subfeature_map[enc_col] = raw_col

def unify_feature_name(name):
    return subfeature_map[name] if name in subfeature_map else name

unified_feature_names = [unify_feature_name(fn) for fn in feature_names]

abs_shap_values = np.abs(shap_values)
raw_feature_aggregation = {}
for i, unified_name in enumerate(unified_feature_names):
    col_mean = abs_shap_values[:, i].mean()
    raw_feature_aggregation[unified_name] = raw_feature_aggregation.get(unified_name, 0.0) + col_mean

shap_df = pd.DataFrame({
    'Feature': list(raw_feature_aggregation.keys()),
    'Mean Absolute SHAP Value': list(raw_feature_aggregation.values())
}).sort_values('Mean Absolute SHAP Value', ascending=False)

plt.figure(figsize=(10, 8))
top_k = 10
plt.barh(shap_df['Feature'].head(top_k), shap_df['Mean Absolute SHAP Value'].head(top_k))
plt.xlabel('Mean Absolute SHAP Value', fontsize=16)
plt.title('Top 10 Most Important Features - GBRT-ECFP Binary Classification', fontsize=16)
plt.gca().invert_yaxis()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# -----------------------------
# B. Grouped Feature Importance
# -----------------------------
group_names = {
    'Photocatalyst Properties': ['Photocatalyst category'],
    'PPCP Properties': ['logP', 'MW'],
    'Membrane Properties': ['Membrane materials', 'Membrane type'],
    'Operational Conditions': [
        'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time', 'Light frequency', 'Hybrid methods'
    ]
}

grouped_shap_values = {}
for group, feats in group_names.items():
    indices = [i for i, ufn in enumerate(unified_feature_names) if ufn in feats]
    if indices:
        sum_per_sample = abs_shap_values[:, indices].sum(axis=1)
        group_mean = sum_per_sample.mean()
    else:
        group_mean = 0.0
    grouped_shap_values[group] = group_mean

grouped_shap_df = pd.DataFrame({
    'Group': list(grouped_shap_values.keys()),
    'Mean Absolute SHAP Value': list(grouped_shap_values.values())
}).sort_values(by='Mean Absolute SHAP Value', ascending=False)

plt.figure(figsize=(10, 5))
plt.barh(grouped_shap_df['Group'], grouped_shap_df['Mean Absolute SHAP Value'], color='C0')
plt.xlabel('Mean Absolute SHAP Value', fontsize=16)
plt.title('GBRT-ECFP Binary Classification', fontsize=16)
plt.gca().invert_yaxis()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# ---------------------------
# C. Feature Importance of ECFP bits specifically
# ---------------------------
ecfp_shap_df = shap_df[shap_df['Feature'].str.startswith('Bit_')].copy()
ecfp_shap_df_sorted = ecfp_shap_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(ecfp_shap_df_sorted['Feature'].head(5),
        ecfp_shap_df_sorted['Mean Absolute SHAP Value'].head(5),
        orientation='vertical')

plt.xlabel('')
plt.xticks([])
plt.ylabel('Mean Absolute SHAP Value', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# ---------------------------
# D. Visualize a specific ECFP bit in a molecule
# ---------------------------
data['ECFP_with_bitinfo'] = data['SMILES'].apply(smiles_to_ecfp_with_bitinfo)

bit_of_interest = 433
print("Bit of interest:", bit_of_interest)

for idx, row in data.iterrows():
    _, bitInfo = row['ECFP_with_bitinfo']
    if bit_of_interest in bitInfo:
        mol = Chem.MolFromSmiles(row['SMILES'])
        display(visualize_substructure(mol, bitInfo, bit_of_interest))
        break
