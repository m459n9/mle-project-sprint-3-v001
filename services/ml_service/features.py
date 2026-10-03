"""Шаги sklearn-пайплайна финальной модели, которые не входят в sklearn.

Модель сохранена через joblib целиком, вместе с этими шагами. При загрузке
pickle ищет классы по пути `ml_service.features.<Класс>`, поэтому модуль нужен
и сервису, и скрипту обучения scripts/train_model.py.

Логика повторяет ноутбук второго спринта (mle-project-sprint-2):
ручные признаки -> ColumnTransformer -> отбор колонок -> CatBoost.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# категориальные признаки модели: код типа дома и булевы флаги
CAT_COLS = ['building_type_int', 'is_apartment', 'studio', 'has_elevator']

# признаки квартиры в том порядке, в котором модель видела их при обучении
INPUT_FEATURES = [
    'floor', 'kitchen_area', 'living_area', 'rooms', 'is_apartment', 'studio',
    'total_area', 'build_year', 'building_type_int', 'latitude', 'longitude',
    'ceiling_height', 'flats_count', 'floors_total', 'has_elevator',
]


class CategoricalToStr(BaseEstimator, TransformerMixin):
    """Приводит категории к строкам: True -> '1', 4 -> '4'.

    Во втором спринте это делалось вручную перед обучением. Здесь шаг встроен
    в пайплайн, чтобы сервис мог отдавать модели значения как есть (bool/int)
    и не было расхождения между обучением и инференсом.
    """

    def __init__(self, cols=tuple(CAT_COLS)):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].astype(int).astype(str)
        return X


class ManualFeatures(BaseEstimator, TransformerMixin):
    """Ручные признаки по гипотезам из EDA: планировка, этаж, дом, локация.

    На fit запоминает «текущий год» (самый новый дом в train) и центр города
    (медианная точка train), как в ноутбуке второго спринта.
    """

    def fit(self, X, y=None):
        self.current_year_ = int(X['build_year'].max())
        self.center_lat_ = float(X['latitude'].median())
        self.center_lon_ = float(X['longitude'].median())
        return self

    def transform(self, X):
        df = X.copy()

        # планировка: сколько метров на комнату и как площадь поделена между зонами
        df['area_per_room'] = df['total_area'] / df['rooms'].clip(lower=1)
        df['living_ratio'] = df['living_area'] / df['total_area']
        df['kitchen_ratio'] = df['kitchen_area'] / df['total_area']
        df['other_area'] = (df['total_area'] - df['living_area'] - df['kitchen_area']).clip(lower=0)
        df['volume'] = df['total_area'] * df['ceiling_height']

        # этаж: первый и последний традиционно дешевле, важна и относительная высота
        df['floor_ratio'] = df['floor'] / df['floors_total'].clip(lower=1)
        df['is_first_floor'] = (df['floor'] == 1).astype(int)
        df['is_last_floor'] = (df['floor'] == df['floors_total']).astype(int)

        # дом: возраст и насколько он «плотный»
        df['building_age'] = self.current_year_ - df['build_year']
        df['flats_per_floor'] = df['flats_count'] / df['floors_total'].clip(lower=1)

        # локация: расстояние до центра в километрах
        lat_km = (df['latitude'] - self.center_lat_) * 111.0
        lon_km = (df['longitude'] - self.center_lon_) * 111.0 * np.cos(np.radians(self.center_lat_))
        df['dist_to_center'] = np.sqrt(lat_km ** 2 + lon_km ** 2)

        return df


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Оставляет колонки матрицы признаков по индексам.

    Аналог mlxtend.feature_selection.ColumnSelector из второго спринта:
    своя реализация в пару строк избавляет сервис от зависимости mlxtend.
    """

    def __init__(self, cols=()):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return X[:, list(self.cols)]
