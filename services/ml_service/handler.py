"""Класс-обработчик запросов к модели оценки стоимости квартир."""
import json
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd

from ml_service.features import INPUT_FEATURES
from ml_service.metrics import (
    INFERENCE_ERRORS,
    INFERENCE_TIME,
    OUT_OF_RANGE,
    OUT_OF_RANGE_REQUESTS,
    PREDICTION,
    PRICE_PER_SQM,
)
from ml_service.schemas import FLAT_EXAMPLE, FlatFeatures

logger = logging.getLogger('uvicorn.error')

# services/ - относительно неё ищутся модель и метаданные
SERVICES_DIR = Path(__file__).resolve().parents[1]


def resolve_path(env_var: str, default: str) -> Path:
    """Путь из переменной окружения или по умолчанию; относительный - от services/."""
    path = Path(os.getenv(env_var, default))
    return path if path.is_absolute() else SERVICES_DIR / path


class FastApiHandler:
    """Загружает модель, проверяет входные данные и возвращает предсказание.

    Ожидаемый формат ответа: {"user_id": <str>, "prediction": <цена, руб.>}.
    """

    def __init__(self, model_path: Path | None = None, meta_path: Path | None = None):
        self.model_path = model_path or resolve_path('MODEL_PATH', 'models/price_model.pkl')
        self.meta_path = meta_path or resolve_path('MODEL_META_PATH', 'models/model_meta.json')
        self.load_model()

    def load_model(self) -> None:
        """Читает пайплайн модели и её метаданные (версия, диапазоны признаков в train)."""
        self.model = joblib.load(self.model_path)
        with open(self.meta_path, encoding='utf-8') as f:
            self.meta = json.load(f)
        self.train_ranges = self.meta.get('train_ranges', {})
        logger.info('модель %s v%s загружена из %s',
                    self.meta.get('model_name'), self.meta.get('model_version'), self.model_path)

    @staticmethod
    def validate_user_id(user_id) -> str:
        """user_id - непустая строка (число тоже подойдёт, приведём к строке)."""
        if user_id is None or not str(user_id).strip():
            raise ValueError('user_id не должен быть пустым')
        return str(user_id).strip()

    @staticmethod
    def validate_params(params: dict | FlatFeatures) -> FlatFeatures:
        """Проверяет признаки квартиры по схеме FlatFeatures.

        Из FastAPI приходит уже проверенный объект, из Python-кода - словарь:
        тогда при ошибке будет pydantic.ValidationError с описанием каждого поля.
        """
        if isinstance(params, FlatFeatures):
            return params
        return FlatFeatures.model_validate(params)

    def check_ranges(self, flat: FlatFeatures) -> list[str]:
        """Признаки вне диапазона обучающей выборки: на них модель экстраполирует."""
        out_of_range = []
        for feature, bounds in self.train_ranges.items():
            value = getattr(flat, feature)
            if value < bounds['min'] or value > bounds['max']:
                out_of_range.append(feature)
                OUT_OF_RANGE.labels(feature=feature).inc()
        if out_of_range:
            OUT_OF_RANGE_REQUESTS.inc()
        return out_of_range

    def predict(self, flat: FlatFeatures) -> float:
        """Цена квартиры в рублях."""
        model_input = pd.DataFrame([flat.model_dump()], columns=INPUT_FEATURES)
        start = time.perf_counter()
        try:
            prediction = float(self.model.predict(model_input)[0])
        except Exception:
            INFERENCE_ERRORS.inc()
            raise
        finally:
            INFERENCE_TIME.observe(time.perf_counter() - start)

        PREDICTION.observe(prediction)
        PRICE_PER_SQM.observe(prediction / flat.total_area)
        return prediction

    def handle(self, user_id, params: dict | FlatFeatures) -> dict:
        """Полный цикл обработки: валидация -> проверка диапазонов -> предсказание."""
        user_id = self.validate_user_id(user_id)
        flat = self.validate_params(params)

        out_of_range = self.check_ranges(flat)
        if out_of_range:
            logger.warning('user_id=%s: признаки вне диапазона train: %s', user_id, out_of_range)

        prediction = self.predict(flat)
        return {'user_id': user_id, 'prediction': round(prediction, 2)}


if __name__ == '__main__':
    # быстрая проверка без веб-сервера: python -m ml_service.handler (из services/)
    logging.basicConfig(level=logging.INFO)
    handler = FastApiHandler()
    print(handler.handle('123', FLAT_EXAMPLE))
