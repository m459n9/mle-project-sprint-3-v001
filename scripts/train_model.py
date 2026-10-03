"""Воспроизводит финальную модель второго спринта и сохраняет её для сервиса.

Финальная модель (версия 4 `flats_price_model` в MLflow Model Registry) обучалась
в ноутбуке mle-project-sprint-2. Скрипт повторяет тот же пайплайн с теми же
отобранными признаками и гиперпараметрами Optuna на том же разбиении
train/test (test_size=0.2, random_state=42) и кладёт результат в services/models:

* price_model.pkl  - sklearn-пайплайн целиком (joblib);
* model_meta.json  - метрики на тесте, гиперпараметры и диапазоны признаков
                     в train (по ним сервис замечает входы вне опыта модели).

Данные - выгрузка таблицы clean_flats_dataset (82 877 строк), например
data/initial_data.csv из DVC-пайплайна первого спринта:

    python scripts/train_model.py --data path/to/initial_data.csv

Для запуска сервиса скрипт не нужен: готовая модель уже лежит в services/models.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import catboost
import joblib
import pandas as pd
import sklearn
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, PolynomialFeatures, StandardScaler

# классы шагов пайплайна должны сохраниться в pickle как ml_service.features.*,
# поэтому импортируем их из пакета сервиса
SERVICES_DIR = Path(__file__).resolve().parents[1] / 'services'
sys.path.insert(0, str(SERVICES_DIR))
from ml_service.features import (  # noqa: E402
    CAT_COLS,
    INPUT_FEATURES,
    CategoricalToStr,
    ColumnSelector,
    ManualFeatures,
)

MODELS_DIR = SERVICES_DIR / 'models'
MODEL_FILE = 'price_model.pkl'
META_FILE = 'model_meta.json'

TARGET_COL = 'price'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# преобразования из этапа 3 второго спринта
POLY_COLS = ['total_area', 'living_area', 'kitchen_area', 'rooms', 'ceiling_height']
BIN_COLS = ['build_year', 'latitude', 'longitude', 'dist_to_center']
MANUAL_FEATURES = [
    'area_per_room', 'living_ratio', 'kitchen_ratio', 'other_area', 'volume',
    'floor_ratio', 'is_first_floor', 'is_last_floor',
    'building_age', 'flats_per_floor', 'dist_to_center',
]
SCALE_COLS = [c for c in INPUT_FEATURES + MANUAL_FEATURES if c not in CAT_COLS + POLY_COLS]

# 11 признаков, которые оставил forward selection (этап 4)
SELECTED_FEATURES = [
    'num__dist_to_center',
    'num__volume',
    'num__latitude',
    'num__longitude',
    'poly__total_area rooms',
    'poly__total_area ceiling_height',
    'poly__total_area kitchen_area',
    'bins__latitude',
    'poly__rooms ceiling_height',
    'bins__build_year',
    'num__floor',
]

# лучшие гиперпараметры Optuna (этап 5)
BEST_PARAMS = {
    'loss_function': 'RMSE',
    'iterations': 1200,
    'learning_rate': 0.02169924677745687,
    'depth': 9,
    'l2_leaf_reg': 1.0778717335941141,
    'min_data_in_leaf': 38,
    'verbose': 0,
    'thread_count': -1,
    'random_seed': RANDOM_STATE,
}

# числовые признаки, для которых сервис следит за выходом за диапазон train
RANGE_FEATURES = [
    'floor', 'kitchen_area', 'living_area', 'rooms', 'total_area', 'build_year',
    'latitude', 'longitude', 'ceiling_height', 'flats_count', 'floors_total',
]


def make_preprocessor():
    return ColumnTransformer([
        ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False), CAT_COLS),
        ('poly', PolynomialFeatures(degree=2, include_bias=False), POLY_COLS),
        # quantile_method='linear' - поведение sklearn 1.3 из второго спринта
        ('bins', KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile',
                                  quantile_method='linear'), BIN_COLS),
        ('num', StandardScaler(), SCALE_COLS),
    ])


def eval_metrics(y_true, y_pred):
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(mean_squared_error(y_true, y_pred)) ** 0.5,
        'mape': float(mean_absolute_percentage_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='csv с таблицей clean_flats_dataset')
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    train, test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    X_train, y_train = train[INPUT_FEATURES], train[TARGET_COL]
    X_test, y_test = test[INPUT_FEATURES], test[TARGET_COL]
    print(f'train: {X_train.shape}, test: {X_test.shape}')

    # индексы отобранных колонок считаем по именам признаков после ColumnTransformer
    prep = Pipeline([
        ('to_str', CategoricalToStr()),
        ('manual', ManualFeatures()),
        ('preprocessor', make_preprocessor()),
    ]).fit(X_train)
    feature_names = list(prep['preprocessor'].get_feature_names_out())
    selected_idx = tuple(feature_names.index(f) for f in SELECTED_FEATURES)

    pipeline = Pipeline([
        ('to_str', CategoricalToStr()),
        ('manual', ManualFeatures()),
        ('preprocessor', make_preprocessor()),
        ('selector', ColumnSelector(cols=selected_idx)),
        ('model', CatBoostRegressor(**BEST_PARAMS)),
    ])
    pipeline.fit(X_train, y_train)

    metrics = eval_metrics(y_test, pipeline.predict(X_test))
    print('метрики на тесте:', {k: round(v, 4) for k, v in metrics.items()})

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / MODEL_FILE)

    meta = {
        'model_name': 'flats_price_model',
        'model_version': '4',
        'source': 'mle-project-sprint-2, MLflow run final_model',
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'library_versions': {'scikit-learn': sklearn.__version__, 'catboost': catboost.__version__},
        'input_features': INPUT_FEATURES,
        'selected_features': SELECTED_FEATURES,
        'params': BEST_PARAMS,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'test_metrics': metrics,
        'train_ranges': {
            col: {'min': float(X_train[col].min()), 'max': float(X_train[col].max())}
            for col in RANGE_FEATURES
        },
    }
    with open(MODELS_DIR / META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'модель: {MODELS_DIR / MODEL_FILE}')
    print(f'метаданные: {MODELS_DIR / META_FILE}')


if __name__ == '__main__':
    main()
