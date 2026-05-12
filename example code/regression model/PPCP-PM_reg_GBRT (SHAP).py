# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:11:02 2024

@author: wangx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
import shap  # Import SHAP library

# 1. Read and prepare data
data = pd.read_csv('PPCP_PM_20251218.csv', encoding='cp1252')
data = data.dropna(how="all").copy()

# Define numeric columns (ensure these match your CSV)
numeric_cols = [
        'logP', 'MW', 'Original concentration (mg/L)', 'pH', 'Dark time', 'Light time'
]

# Define raw categorical columns to be one-hot encoded
categorical_cols = [
    'Photocatalyst category',
    'Membrane materials',
    'Membrane type',
    'Light frequency',
    'Hybrid methods'
]

# 2. One-hot encode the categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = ohe.fit_transform(data[categorical_cols])
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)

# 3. Keep numeric and categorical parts separate (for numeric-only scaling)
features_numeric = data[numeric_cols].values

# 4. Define the target variable
targets = data['Removal efficiency'].values

# 5. NOTE: Do NOT scale the full feature matrix here (avoids leakage).
# Numeric-only scaling will be done inside each CV fold.

# 6. Setup K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# 7. Gradient Boosting parameter sets (include your original as Option A)
gbr_param_list = [
    # (A) original (random_state removed per preference)
    {
        'n_estimators': 400,
        'learning_rate': 0.01,
        'max_depth': 9,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 0.8
    },
    # (B) more trees, slower learning, shallower depth (more regularized)
    {
        'n_estimators': 400,
        'learning_rate': 0.001,
        'max_depth': 9,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 0.8
    },
    # (C) fewer trees, faster learning, moderate depth
    {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 0.8
    },
]


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

# 8. Perform multiple runs
n_runs = 5
run_results = []
for run in range(n_runs):
    all_measured = []
    all_predicted = []
    print(f"\n=== Run {run+1} ===")
    
    # Split indices using numeric part (same length as full dataset)
    for train_index, test_index in kf.split(features_numeric):

        # ------------------------------------------------
        # Fold-safe numeric-only scaling (no leakage)
        # ------------------------------------------------
        Xnum_train_raw, Xnum_test_raw = features_numeric[train_index], features_numeric[test_index]
        Xcat_train,     Xcat_test     = cat_features[train_index],     cat_features[test_index]

        train_targets_full, test_targets = targets[train_index], targets[test_index]

        # Fit scaler ONLY on numeric training fold
        scaler_fold = StandardScaler()
        Xnum_train = scaler_fold.fit_transform(Xnum_train_raw)
        Xnum_test  = scaler_fold.transform(Xnum_test_raw)

        # Recombine: scaled numeric + unscaled one-hot categorical
        train_features_full = np.hstack((Xnum_train, Xcat_train))
        test_features       = np.hstack((Xnum_test,  Xcat_test))
        # ------------------------------------------------

        # 90/10 validation split inside the training fold
        cut = int(len(train_features_full) * 0.9)
        X_tr, y_tr = train_features_full[:cut], train_targets_full[:cut]
        X_val, y_val = train_features_full[cut:], train_targets_full[cut:]

        # select best params by validation R^2
        best_score = -np.inf
        best_params = None
        for p in gbr_param_list:
            model = GradientBoostingRegressor(**p)
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            if val_r2 > best_score:
                best_score = val_r2
                best_params = p
                
        # Print the best parameters for this fold
        print(best_params)
        
        # retrain best model on full training (train + val)
        model = GradientBoostingRegressor(**best_params)
        model.fit(train_features_full, train_targets_full)
        
        # Predict and collect data
        predictions = model.predict(test_features)
        all_measured.extend(test_targets)
        all_predicted.extend(predictions)
        
    # Calculate metrics for the run
    mse = mean_squared_error(all_measured, all_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_measured, all_predicted)
    r2 = r2_score(all_measured, all_predicted)
    run_results.append((r2, mae, rmse, mse))

    # Print metrics for this run (unchanged format)
    print(f"Run {run+1} - R2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}, MSE: {mse:.3f}")
    
    # Plot aggregated predictions for the run (unchanged style)
    plt.figure(figsize=(6, 4))
    plt.scatter(all_measured, all_predicted, c='blue', alpha=0.5, label='Predictions')
    plt.plot([min(all_measured), max(all_measured)], [min(all_measured), max(all_measured)], 'r-', label='Ideal')
    plt.xlabel('Measured Efficiency', fontsize=12)
    plt.ylabel('Predicted Efficiency', fontsize=12)
    plt.title(f'$R^2$ = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}, GBRT', fontsize=11)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    pooled_measured.extend(all_measured)
    pooled_predicted.extend(all_predicted)

# 9. Calculate overall average and std of metrics
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
# 9B. Error distribution analysis for the best GBRT model
#     based on pooled out-of-fold predictions
# ------------------------------------------------
pooled_measured_arr = np.array(pooled_measured)
pooled_predicted_arr = np.array(pooled_predicted)

residuals = pooled_predicted_arr - pooled_measured_arr
absolute_errors = np.abs(residuals)

# Residual histogram
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Residual (Predicted - Measured)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Error Distribution of Best GBRT Model: Residuals', fontsize=11)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("Figure_S2a_residual_histogram.png", dpi=300, bbox_inches='tight')
plt.show()

# Absolute error histogram
plt.figure(figsize=(6, 4))
plt.hist(absolute_errors, bins=30, edgecolor='black')
plt.xlabel('Absolute Error |Predicted - Measured|', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Error Distribution of Best GBRT Model: Absolute Errors', fontsize=11)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("Figure_S2b_absolute_error_histogram.png", dpi=300, bbox_inches='tight')
plt.show()
# ------------------------------------------------
# 10. Feature Importance Section
# ------------------------------------------------

# Build full-dataset features for final training / SHAP / PDP (numeric-only scaling)
scaler_full = StandardScaler()
features_numeric_scaled = scaler_full.fit_transform(features_numeric)
features_scaled = np.hstack((features_numeric_scaled, cat_features))

# Retrain the model on the entire dataset
model.fit(features_scaled, targets)

# SHAP Explainer & Values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features_scaled)

# Combine numeric and encoded feature names
feature_names = numeric_cols + list(encoded_feature_names)

# -----------------------------
# A. Aggregated Feature-Level SHAP (Top-20 Chart)
# -----------------------------
# 1. Build a map from each encoded col -> raw categorical col
subfeature_map = {}
for raw_col in categorical_cols:
    for enc_col in encoded_feature_names:
        if enc_col.startswith(raw_col + "_"):
            subfeature_map[enc_col] = raw_col
        elif enc_col == raw_col:
            subfeature_map[enc_col] = raw_col

# 2. unify_feature_name
def unify_feature_name(name):
    if name in subfeature_map:
        return subfeature_map[name]
    else:
        return name

# 3. For each column in feature_names, unify if it's a sub-feature
unified_feature_names = [unify_feature_name(fn) for fn in feature_names]

# 4. Sum absolute shap values by raw column
abs_shap_values = np.abs(shap_values)
M = len(feature_names)
raw_feature_aggregation = {}

for i in range(M):
    disp_name = unified_feature_names[i]
    # Mean absolute shap for sub-feature i
    col_mean = abs_shap_values[:, i].mean()
    if disp_name not in raw_feature_aggregation:
        raw_feature_aggregation[disp_name] = 0.0
    raw_feature_aggregation[disp_name] += col_mean

# 5. Build a DataFrame & plot top 20
shap_df_agg = pd.DataFrame({
    'Feature': list(raw_feature_aggregation.keys()),
    'Mean Absolute SHAP Value': list(raw_feature_aggregation.values())
}).sort_values('Mean Absolute SHAP Value', ascending=False)

plt.figure(figsize=(10, 8))
top_k = 10
plt.barh(shap_df_agg['Feature'].head(top_k), shap_df_agg['Mean Absolute SHAP Value'].head(top_k))
plt.xlabel('Mean Absolute SHAP Value', fontsize=16)
plt.title('Top 10 Most Important Features - GBRT Regression', fontsize=16)
plt.gca().invert_yaxis()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# -----------------------------
# B. Grouped Feature Importance
# -----------------------------
group_names = {
    'Photocatalyst Properties': [
        'Photocatalyst category'
    ],
    'PPCP Properties': ['logP', 'MW'],
    'Membrane Properties': ['Membrane materials', 'Membrane type'],
    'Operational Conditions': [
        'Original concentration (mg/L)', 'pH','Dark time', 'Light time', 'Light frequency', 'Hybrid methods'
    ]
}

grouped_shap_values = {}
for group, feats in group_names.items():
    # Indices for sub-features that unify to any raw feature in feats
    indices = []
    for i, ufn in enumerate(unified_feature_names):
        if ufn in feats:
            indices.append(i)
    
    if indices:
        # Sum absolute shap across sub-features, then average across samples
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
plt.title('GBRT Regression', fontsize=16)
plt.gca().invert_yaxis()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# ------------------------------------------------
# 11. 2D PDP (Categorical × Numeric) — manual, robust
#     Photocatalyst category × Light time
#     - log x-axis
#     - PDP computed on log-spaced x_grid from 0.5 to 1200 h
# ------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FixedLocator, FuncFormatter

# Reuse the same transformer (raw df -> model input)
def data_transformer(df_in):
    Xnum = scaler_full.transform(df_in[numeric_cols].values)
    Xcat = ohe.transform(df_in[categorical_cols])
    return np.hstack((Xnum, Xcat))

# -------------------------
# A) Choose grids (LOG grid)
# -------------------------
x_min, x_max = 3, 2400.0
n_x = 25
x_grid = np.geomspace(x_min, x_max, n_x)  # log-spaced grid

pcat_col = "Photocatalyst category"
# stable order (alphabetical); replace with a custom list if desired
y_labels = sorted(data[pcat_col].dropna().unique())
y_grid = np.arange(len(y_labels))

# Optional subsampling for speed (PDP becomes an approximation if sampled)
max_rows = 3000
data_base = data.sample(n=max_rows, random_state=0) if len(data) > max_rows else data.copy()

# -------------------------
# B) Compute PDP surface Z
# -------------------------
Z = np.zeros((len(y_labels), len(x_grid)), dtype=float)

for i, cat in enumerate(y_labels):
    df_cat = data_base.copy()
    df_cat[pcat_col] = cat

    for j, x in enumerate(x_grid):
        df_ij = df_cat.copy()
        df_ij["Light time"] = float(x)

        X_ij = data_transformer(df_ij)
        Z[i, j] = model.predict(X_ij).mean()

# -------------------------
# Helper: build log-safe x-edges for pcolormesh
# -------------------------
def log_edges_from_centers(xc):
    xc = np.asarray(xc, dtype=float)
    x_edges = np.empty(len(xc) + 1, dtype=float)
    # internal edges: geometric mean between adjacent centers
    x_edges[1:-1] = np.sqrt(xc[:-1] * xc[1:])
    # boundary edges: extrapolate in log space
    x_edges[0] = xc[0] / np.sqrt(xc[1] / xc[0])
    x_edges[-1] = xc[-1] * np.sqrt(xc[-1] / xc[-2])
    return x_edges

# tick formatter: show clean numbers on log scale
def nice_log_tick(v, pos=None):
    # show as integer if >= 1, else keep one decimal
    if v >= 1:
        return f"{int(v)}" if abs(v - int(v)) < 1e-9 else f"{v:g}"
    return f"{v:.1f}"

# choose “reader-friendly” log ticks
xticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

# -------------------------
# C) Plot 1: all categories
# -------------------------
fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=300)

y_edges = np.arange(len(y_labels) + 1) - 0.5
x_edges = log_edges_from_centers(x_grid)

vmin, vmax = 0.25, 0.85  # adjust to your global plan

pc = ax.pcolormesh(
    x_edges, y_edges, Z,
    cmap="viridis",
    shading="flat",
    vmin=vmin, vmax=vmax
)

for yline in range(len(y_labels)):
    ax.axhline(yline + 0.5, color="white", linewidth=0.6, alpha=0.6)

ax.set_yticks(y_grid)
ax.set_yticklabels(y_labels, fontsize=12)

# log x-axis
ax.set_xscale("log")
ax.set_xlim(x_min, x_max)
ax.xaxis.set_major_locator(FixedLocator(xticks))
ax.xaxis.set_major_formatter(FuncFormatter(nice_log_tick))
ax.tick_params(axis="x", labelsize=10)

ax.set_xlabel("Light time (h)", fontsize=11)
ax.set_ylabel("Photocatalyst category", fontsize=11)
ax.set_title("2D PDP: Photocatalyst category × Light time", fontsize=11)

cbar = fig.colorbar(pc, ax=ax, pad=0.02)
cbar.set_label("Partial dependence", fontsize=11)
cbar.ax.tick_params(labelsize=11)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.set_ylim(-0.5, len(y_labels) - 0.5)
ax.grid(False)

plt.tight_layout()
plt.show()


# ------------------------------------------------
# 12. 2D PDP (Categorical × Numeric) by definition (manual):
#     Membrane materials × logP
#     (same heatmap method as Photocatalyst category × Light time)
# ------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Reuse the same transformer
def data_transformer(df_in):
    Xnum = scaler_full.transform(df_in[numeric_cols].values)
    Xcat = ohe.transform(df_in[categorical_cols])
    return np.hstack((Xnum, Xcat))

# --- A) Choose grids ---
# logP grid (percentile grid similar to your earlier style)
x_raw = data["logP"].values.astype(float)
n_x = 25  # match the visual density you used before (e.g., 25)
pct = np.linspace(0, 100, n_x)
x_grid = np.unique(np.percentile(x_raw, pct))

# membrane categories (same order as OneHotEncoder)
mem_idx = categorical_cols.index("Membrane materials")
y_labels = list(ohe.categories_[mem_idx])
y_grid = np.arange(len(y_labels))

# Optional: downsample rows for speed (keeps PDP shape, faster)
# If dataset is small, this keeps everything.
max_rows = 3000
if len(data) > max_rows:
    data_base = data.sample(n=max_rows).copy()
else:
    data_base = data.copy()

# --- B) Compute PDP surface Z (ny × nx) ---
Z = np.zeros((len(y_labels), len(x_grid)), dtype=float)

for i, mem in enumerate(y_labels):
    df_mem = data_base.copy()
    df_mem["Membrane materials"] = mem

    for j, x in enumerate(x_grid):
        df_ij = df_mem.copy()
        df_ij["logP"] = x

        X_ij = data_transformer(df_ij)
        Z[i, j] = model.predict(X_ij).mean()

# --- C) Plot (publication style): heatmap tiles ---
fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=300)

# y edges (categorical bands)
y_edges = np.arange(len(y_labels) + 1) - 0.5

# x edges (grid midpoints)
x_edges = np.empty(len(x_grid) + 1)
x_edges[1:-1] = 0.5 * (x_grid[:-1] + x_grid[1:])
x_edges[0] = x_grid[0] - 0.5 * (x_grid[1] - x_grid[0])
x_edges[-1] = x_grid[-1] + 0.5 * (x_grid[-1] - x_grid[-2])

# Use the same vmin/vmax across categorical×numeric PDPs for comparability
vmin, vmax = 0.55, 0.85

pc = ax.pcolormesh(
    x_edges, y_edges, Z,
    cmap="viridis",
    shading="flat",
    vmin=vmin,
    vmax=vmax
)

# subtle separators between categories
for yline in range(len(y_labels)):
    ax.axhline(yline + 0.5, color="white", linewidth=0.6, alpha=0.6)

# y-axis labels
ax.set_yticks(y_grid)
ax.set_yticklabels(y_labels, fontsize=11)

# ---- Make logP axis reader-friendly ----
xticks = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

ax.set_xticks(xticks)
ax.set_xticklabels([f"{v:}" if v != 0 else "0" for v in xticks], fontsize=11)

# Optional: ensure full range is shown
ax.set_xlim(min(xticks), max(xticks))


ax.set_xlabel("logP", fontsize=11)
ax.set_ylabel("Membrane materials", fontsize=11)
ax.set_title("2D PDP: Membrane materials × logP", fontsize=11)

cbar = fig.colorbar(pc, ax=ax, pad=0.02)
cbar.set_label("Partial dependence", fontsize=11)
cbar.ax.tick_params(labelsize=11)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.set_ylim(-0.5, len(y_labels) - 0.5)
ax.grid(False)

plt.tight_layout()
plt.show()
