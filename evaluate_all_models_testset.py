"""
Evaluate Linear Regression, Random Forest, and XGBoost on the FD001 dataset.

Pipeline:
1. Validation: GroupShuffleSplit by engine unit (no data leakage)
2. Test set: retrain all models on FULL training data, evaluate on test_FD001
   last-cycle rows vs RUL_FD001.txt ground truth
3. Generate figures: pred_vs_actual and feature_importance (300 DPI)

Preprocessing matches notebook 01 exactly:
- Same column names, same constant-column removal (variance < 1e-6)
- Same RUL clipping at 125
- Same MinMaxScaler fitted on full training data
- Same model hyperparameters and random seeds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style('whitegrid')

# ── 1. Load data ──────────────────────────────────────────────────────────────
columns = ['unit', 'cycle',
           'setting_1', 'setting_2', 'setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]

train_df = pd.read_csv('CMaps/train_FD001.txt',
                        sep=r'\s+', header=None, names=columns)
test_df  = pd.read_csv('CMaps/test_FD001.txt',
                        sep=r'\s+', header=None, names=columns)
rul_true = pd.read_csv('CMaps/RUL_FD001.txt',
                        header=None, names=['RUL'])

# ── 2. Construct RUL labels ──────────────────────────────────────────────────
max_cycles = train_df.groupby('unit')['cycle'].max()
train_df['RUL'] = train_df.apply(
    lambda row: max_cycles[row['unit']] - row['cycle'], axis=1)

RUL_CLIP = 125
train_df['RUL_clipped'] = train_df['RUL'].clip(upper=RUL_CLIP)

# ── 3. Feature selection: drop constant columns ─────────────────────────────
sensor_cols  = [f'sensor_{i}' for i in range(1, 22)]
feature_cols = ['setting_1', 'setting_2', 'setting_3'] + sensor_cols
variances    = train_df[feature_cols].var()
constant_cols = variances[variances < 1e-6].index.tolist()
feature_cols  = [c for c in feature_cols if c not in constant_cols]

print(f"Features used ({len(feature_cols)}): {feature_cols}")
print(f"Constant cols dropped: {constant_cols}")

# ── 4. Normalize features (fit on full training data) ────────────────────────
scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

# ── 5. Validation: GroupShuffleSplit by engine unit ──────────────────────────
X_all = train_df[feature_cols].values
y_all = train_df['RUL_clipped'].values
groups = train_df['unit'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X_all, y_all, groups))

X_train_split, X_val = X_all[train_idx], X_all[val_idx]
y_train_split, y_val = y_all[train_idx], y_all[val_idx]

train_units = set(groups[train_idx])
val_units   = set(groups[val_idx])
print(f"\n── Validation Split (by engine unit) ──")
print(f"Training: {len(train_units)} engines, {len(train_idx)} samples")
print(f"Validation: {len(val_units)} engines, {len(val_idx)} samples")
print(f"Overlap check: {train_units & val_units} (should be empty)")

# ── 6. Validation results ───────────────────────────────────────────────────
model_specs = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest',     RandomForestRegressor(n_estimators=100, random_state=42)),
    ('XGBoost',           XGBRegressor(n_estimators=100, learning_rate=0.1,
                                       random_state=42)),
]

print(f"\n── Validation Set Results (grouped by unit, no leakage) ──")
print(f"{'Model':20s} | {'RMSE':>7s} | {'MAE':>7s} | {'R²':>7s}")
print('-' * 52)

val_results = []
for name, model in model_specs:
    model.fit(X_train_split, y_train_split)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae  = mean_absolute_error(y_val, y_pred)
    r2   = r2_score(y_val, y_pred)
    val_results.append({'name': name, 'rmse': rmse, 'mae': mae, 'r2': r2})
    print(f"{name:20s} | {rmse:7.2f} | {mae:7.2f} | {r2:7.3f}")

# ── 7. Test set: retrain on FULL training data ──────────────────────────────
test_last = test_df.groupby('unit').last().reset_index()
test_last[feature_cols] = scaler.transform(test_last[feature_cols])
X_test = test_last[feature_cols].values
y_test_true = rul_true['RUL'].clip(upper=RUL_CLIP).values

# Fresh model instances with same hyperparameters
model_specs_test = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest',     RandomForestRegressor(n_estimators=100, random_state=42)),
    ('XGBoost',           XGBRegressor(n_estimators=100, learning_rate=0.1,
                                       random_state=42)),
]

print(f"\n── Test Set Results (trained on full training data) ──")
print(f"{'Model':20s} | {'RMSE':>7s} | {'MAE':>7s} | {'R²':>7s}")
print('-' * 52)

test_results = []
test_models = {}
for name, model in model_specs_test:
    model.fit(X_all, y_all)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
    mae  = mean_absolute_error(y_test_true, y_pred)
    r2   = r2_score(y_test_true, y_pred)
    test_results.append({'name': name, 'rmse': rmse, 'mae': mae, 'r2': r2})
    test_models[name] = (model, y_pred)
    print(f"{name:20s} | {rmse:7.2f} | {mae:7.2f} | {r2:7.3f}")

# ── 8. Improvement ──────────────────────────────────────────────────────────
lr_rmse  = test_results[0]['rmse']
xgb_rmse = test_results[2]['rmse']
improvement = (lr_rmse - xgb_rmse) / lr_rmse * 100

print(f"\n── Improvement ──")
print(f"XGBoost RMSE vs Linear Regression RMSE (both on test set):")
print(f"  ({lr_rmse:.2f} - {xgb_rmse:.2f}) / {lr_rmse:.2f} = {improvement:.1f}% reduction")

# ── 9. Generate figures ──────────────────────────────────────────────────────
best_name = 'XGBoost'
best_model, y_test_pred = test_models[best_name]

# Figure 1: Predicted vs Actual
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test_true, y_test_pred, alpha=0.6, edgecolors='black', linewidths=0.5)
max_val = max(y_test_true.max(), y_test_pred.max()) + 10
ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
ax.set_xlabel('Actual RUL')
ax.set_ylabel('Predicted RUL')
ax.set_title(f'Predicted vs Actual RUL ({best_name})')
ax.legend()
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
plt.tight_layout()
fig.savefig('results/figures/fig_pred_vs_actual.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/figures/fig_pred_vs_actual.png")
plt.close(fig)

# Figure 2: Feature Importance
importance = pd.Series(best_model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
importance.plot(kind='barh', ax=ax)
ax.set_title(f'Feature Importance ({best_name})')
ax.set_xlabel('Importance')
plt.tight_layout()
fig.savefig('results/figures/fig_feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/fig_feature_importance.png")
plt.close(fig)
