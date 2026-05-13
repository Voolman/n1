import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def prepare_features_extreme(df):
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['METEOFORECASTHOUR_OPENM_Datetime'])
    
    # Time-based features
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)
    
    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Wind Physics
    for h in ['80m', '120m', '180m']:
        if f'wind_speed_{h}' in df.columns:
            df[f'wind_power_{h}'] = df[f'wind_speed_{h}']**3
            # Energy is proportional to v^3, but turbine efficiency changes. 
            # Adding v^2 and sqrt(v) might help the model find the power curve.
            df[f'wind_speed_{h}_sq'] = df[f'wind_speed_{h}']**2
            df[f'wind_speed_{h}_sqrt'] = np.sqrt(df[f'wind_speed_{h}'])
    
    # Wind Shear & Vector components
    for h in ['10m', '80m', '120m', '180m']:
        if f'wind_speed_{h}' in df.columns and f'wind_direction_{h}' in df.columns:
            rad = np.radians(df[f'wind_direction_{h}'])
            df[f'wind_u_{h}'] = df[f'wind_speed_{h}'] * np.cos(rad)
            df[f'wind_v_{h}'] = df[f'wind_speed_{h}'] * np.sin(rad)

    # Temperature and Pressure
    df['temp_diff'] = df['temperature_120m'] - df['temperature_80m']
    df['rho_proxy'] = df['pressure_msl'] / (df['temperature_120m'] + 273.15)
    
    # Interaction
    df['energy_proxy'] = df['wind_power_120m'] * df['rho_proxy']
    
    cols_to_drop = ['METEOFORECASTHOUR_OPENM_Datetime', 'datetime']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Load data
train_df = pd.read_csv('/home/ubuntu/task_data/dataset/train_dataset.csv')
valid_df = pd.read_csv('/home/ubuntu/task_data/dataset/valid_features.csv')

# Handle missing values
train_df['wind_speed_180m'] = train_df['wind_speed_180m'].fillna(train_df['wind_speed_120m'] * 1.05)
train_df['wind_direction_180m'] = train_df['wind_direction_180m'].fillna(train_df['wind_direction_120m'])

target_col = 'Выработка. Результирующий расчет'
y = train_df[target_col]
X = train_df.drop(columns=[target_col])

# Prepare features
X = prepare_features_extreme(X)
X_test = prepare_features_extreme(valid_df)
X_test = X_test[X.columns]

print(f"Total features: {len(X.columns)}")

# 10-Fold CV for maximum stability
N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

oof_cat = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

test_cat = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {i+1}/{N_SPLITS} ---")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # CatBoost: High iterations, low learning rate
    cat = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.02,
        depth=9,
        l2_leaf_reg=4,
        random_seed=RANDOM_SEED,
        verbose=1000,
        early_stopping_rounds=400,
        loss_function='RMSE',
        eval_metric='MAE'
    )
    cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    oof_cat[val_idx] = cat.predict(X_val)
    test_cat += cat.predict(X_test) / N_SPLITS
    
    # LightGBM: Large number of leaves, small learning rate
    lgb = LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='mae', callbacks=[])
    oof_lgb[val_idx] = lgb.predict(X_val)
    test_lgb += lgb.predict(X_test) / N_SPLITS
    
    # XGBoost: Added to ensemble
    xgb = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric='mae',
        early_stopping_rounds=300
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb.predict(X_val)
    test_xgb += xgb.predict(X_test) / N_SPLITS

# Blending optimization
from scipy.optimize import minimize

def objective(w):
    preds = w[0] * oof_cat + w[1] * oof_lgb + w[2] * oof_xgb
    return mean_absolute_error(y, preds)

init_guess = [0.4, 0.3, 0.3]
res = minimize(objective, init_guess, bounds=((0,1), (0,1), (0,1)), constraints={'type': 'eq', 'fun': lambda w: 1 - sum(w)})
best_w = res.x

print(f"\nOptimal Weights: CatBoost={best_w[0]:.3f}, LGBM={best_w[1]:.3f}, XGBoost={best_w[2]:.3f}")
final_mae = objective(best_w)
print(f"Final Optimized MAE: {final_mae:.4f}")

# Final predictions
final_preds = best_w[0] * test_cat + best_w[1] * test_lgb + best_w[2] * test_xgb
pd.DataFrame(final_preds).to_csv('/home/ubuntu/predictions_extreme.csv', index=False, header=False)

print("\nExtreme predictions saved to /home/ubuntu/predictions_extreme.csv")
