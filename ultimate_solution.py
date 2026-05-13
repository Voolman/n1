import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import json
import pickle
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ============================================================
# КОНФИГУРАЦИЯ И ГИПЕРПАРАМЕТРЫ
# ============================================================
RANDOM_SEED = 42
N_FOLDS = 10  # Увеличено для стабильности
TARGET_COL = 'Выработка. Результирующий расчет'
DATETIME_COL = 'METEOFORECASTHOUR_OPENM_Datetime'
INSTALLED_CAPACITY = 90.09
TURBINE_POWER = 3.465
N_TURBINES = 26

# Пути (настройте под свою систему)
TRAIN_PATH = '/home/ubuntu/task_data/dataset/train_dataset.csv'
VALID_PATH = '/home/ubuntu/task_data/dataset/valid_features.csv'
MODEL_DIR = 'ultimate_models'
PREDICTIONS_PATH = 'ultimate_predictions.csv'
FEEDBACK_FILE = 'feedback_metrics.json'

np.random.seed(RANDOM_SEED)

# ============================================================
# FEATURE ENGINEERING (ОБЪЕДИНЕННЫЙ)
# ============================================================
def create_features_ultimate(df, is_train=True):
    df = df.copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    
    # Временные признаки
    df['hour'] = df[DATETIME_COL].dt.hour
    df['month'] = df[DATETIME_COL].dt.month
    df['day_of_year'] = df[DATETIME_COL].dt.dayofyear
    df['day_of_week'] = df[DATETIME_COL].dt.dayofweek
    
    # Циклическое кодирование
    for col, max_val in [('hour', 24), ('month', 12), ('day_of_year', 365)]:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    
    # Физика ветра (v^3, v^2, sqrt(v))
    for h in ['10m', '80m', '120m', '180m']:
        if f'wind_speed_{h}' in df.columns:
            # Обработка пропусков для 180m
            if h == '180m':
                df[h] = df[h].fillna(df['wind_speed_120m'] * 1.05)
            
            df[f'ws_{h}_p3'] = df[f'wind_speed_{h}'] ** 3
            df[f'ws_{h}_p2'] = df[f'wind_speed_{h}'] ** 2
            df[f'ws_{h}_sqrt'] = np.sqrt(df[f'wind_speed_{h}'])
            
            # Векторные компоненты
            if f'wind_direction_{h}' in df.columns:
                if h == '180m':
                    df[f'wind_direction_{h}'] = df[f'wind_direction_{h}'].fillna(df['wind_direction_120m'])
                rad = np.radians(df[f'wind_direction_{h}'])
                df[f'wind_u_{h}'] = df[f'wind_speed_{h}'] * np.cos(rad)
                df[f'wind_v_{h}'] = df[f'wind_speed_{h}'] * np.sin(rad)

    # Wind Shear
    df['shear_120_80'] = df['wind_speed_120m'] - df['wind_speed_80m']
    df['shear_80_10'] = df['wind_speed_80m'] - df['wind_speed_10m']
    
    # Плотность воздуха и энергия
    temp_k = df['temperature_80m'] + 273.15
    df['rho'] = (df['pressure_msl'] * 100) / (287.05 * temp_k)
    df['energy_80m'] = 0.5 * df['rho'] * df['ws_80m_p3']
    df['energy_120m'] = 0.5 * df['rho'] * df['ws_120m_p3']
    
    # Ремонт и мощность
    df['working_turbines'] = N_TURBINES - df['Кол-во_ВЭУ_в_ремонте']
    df['capacity_factor'] = df['working_turbines'] / N_TURBINES
    
    # Удаление лишнего
    drop_cols = [DATETIME_COL, 'hour', 'month', 'day_of_year']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    return df

# ============================================================
# МЕХАНИЗМ ОБРАТНОЙ СВЯЗИ (FEEDBACK LOOP)
# ============================================================
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return {"history": [], "best_error": float('inf'), "adjustment_factor": 1.0}

def save_feedback(error_rate):
    feedback = load_feedback()
    feedback['history'].append(error_rate)
    
    # Если ошибка растет, уменьшаем learning_rate, если падает - пробуем чуть ускорить
    if error_rate < feedback['best_error']:
        feedback['best_error'] = error_rate
        feedback['adjustment_factor'] *= 0.95 # Тонкая подстройка
    else:
        feedback['adjustment_factor'] *= 1.05 # Поиск другого минимума
        
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback, f)
    return feedback['adjustment_factor']

# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ============================================================
def train_ultimate(external_error=None):
    adj_factor = 1.0
    if external_error is not None:
        adj_factor = save_feedback(external_error)
        print(f"\n[!] Получена ошибка с сайта: {external_error}. Коэффициент адаптации: {adj_factor:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    train_raw = pd.read_csv(TRAIN_PATH)
    train_df = create_features_ultimate(train_raw)
    
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    feature_names = X.columns.tolist()
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Хранилища для OOF предсказаний
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    
    # Параметры с учетом адаптации
    lgb_params = {
        'n_estimators': 5000,
        'learning_rate': 0.02 * adj_factor,
        'num_leaves': 127,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'importance_type': 'gain'
    }
    
    cat_params = {
        'iterations': 5000,
        'learning_rate': 0.02 * adj_factor,
        'depth': 9,
        'l2_leaf_reg': 4,
        'random_seed': RANDOM_SEED,
        'loss_function': 'RMSE',
        'eval_metric': 'MAE',
        'early_stopping_rounds': 400,
        'verbose': False
    }

    print(f"\n[1/3] Обучение ансамбля на {N_FOLDS} фолдах...")
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        # LightGBM
        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='mae', callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        oof_lgb[val_idx] = m_lgb.predict(X_val)
        
        # CatBoost
        m_cat = CatBoostRegressor(**cat_params)
        m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof_cat[val_idx] = m_cat.predict(X_val)
        
        # XGBoost
        m_xgb = xgb.XGBRegressor(n_estimators=5000, learning_rate=0.02*adj_factor, max_depth=8, subsample=0.8, colsample_bytree=0.7, random_state=RANDOM_SEED, eval_metric='mae', early_stopping_rounds=300)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = m_xgb.predict(X_val)
        
        print(f"  Fold {fold+1} complete.")

    # Оптимизация весов ансамбля (Blending)
    def objective(w):
        preds = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        return mean_absolute_error(y, preds)

    res = minimize(objective, [0.33, 0.33, 0.34], bounds=((0,1), (0,1), (0,1)), constraints={'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    weights = res.x
    
    print(f"\n[2/3] Оптимальные веса: LGB={weights[0]:.3f}, Cat={weights[1]:.3f}, XGB={weights[2]:.3f}")
    print(f"MAE Ансамбля на CV: {objective(weights):.4f}")

    # Финальное предсказание
    print("\n[3/3] Генерация финальных предсказаний...")
    valid_raw = pd.read_csv(VALID_PATH)
    valid_df = create_features_ultimate(valid_raw)
    X_test = valid_df[feature_names]
    
    # Для простоты в этом скрипте обучаем финальные модели на всем train
    # В идеале - использовать усреднение моделей с фолдов
    m_lgb_final = lgb.LGBMRegressor(**lgb_params).fit(X, y)
    m_cat_final = CatBoostRegressor(**cat_params).fit(X, y)
    m_xgb_final = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02*adj_factor, max_depth=8).fit(X, y)
    
    p_lgb = m_lgb_final.predict(X_test)
    p_cat = m_cat_final.predict(X_test)
    p_xgb = m_xgb_final.predict(X_test)
    
    final_preds = weights[0] * p_lgb + weights[1] * p_cat + weights[2] * p_xgb
    final_preds = np.clip(final_preds, 0, INSTALLED_CAPACITY)
    
    pd.DataFrame(final_preds).to_csv(PREDICTIONS_PATH, index=False, header=False)
    print(f"Предсказания сохранены в {PREDICTIONS_PATH}")

if __name__ == '__main__':
    import sys
    err = None
    if len(sys.argv) > 1:
        try:
            err = float(sys.argv[1])
        except:
            pass
    train_ultimate(err)
