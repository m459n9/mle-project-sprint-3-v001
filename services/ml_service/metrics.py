"""Собственные метрики сервиса (prometheus_client).

Базовые HTTP-метрики (число запросов, коды ответов, время ответа) отдаёт
prometheus_fastapi_instrumentator, здесь - то, чего он не знает: что модель
предсказывает, сколько думает и какие данные к ней приходят.
Все метрики попадают в общий реестр и отдаются на /metrics.
"""
from prometheus_client import Counter, Histogram

# цены в датасете - от 1 до 21 млн руб., медиана 10.5 млн
PREDICTION = Histogram(
    'price_model_prediction_rub',
    'Предсказанная цена квартиры, руб.',
    buckets=(3e6, 5e6, 7e6, 9e6, 11e6, 13e6, 15e6, 18e6, 21e6, 25e6, 30e6),
)

# медиана цены за метр в train - около 220 тыс. руб.
PRICE_PER_SQM = Histogram(
    'price_model_price_per_sqm_rub',
    'Предсказанная цена за квадратный метр, руб.',
    buckets=(100e3, 150e3, 175e3, 200e3, 225e3, 250e3, 275e3, 300e3, 350e3, 400e3, 500e3),
)

INFERENCE_TIME = Histogram(
    'price_model_inference_seconds',
    'Время предсказания модели (без сети и валидации), с',
    buckets=(0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)

# признак в допустимых границах, но вне диапазона train: модель экстраполирует
OUT_OF_RANGE = Counter(
    'price_model_out_of_range_total',
    'Признаки запроса за пределами диапазона обучающей выборки',
    ['feature'],
)

# запросы, где вне диапазона train хотя бы один признак; доля от всех предсказаний -
# главный сигнал дрейфа входных данных
OUT_OF_RANGE_REQUESTS = Counter(
    'price_model_out_of_range_requests_total',
    'Запросы, в которых хотя бы один признак вне диапазона обучающей выборки',
)

# по какому полю чаще всего падает валидация - подсказка, что сломалось у клиента
VALIDATION_ERRORS = Counter(
    'price_model_validation_errors_total',
    'Запросы, отклонённые валидацией, по полям с ошибкой',
    ['field'],
)

INFERENCE_ERRORS = Counter(
    'price_model_inference_errors_total',
    'Ошибки модели при предсказании',
)
