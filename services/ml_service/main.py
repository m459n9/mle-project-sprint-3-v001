"""FastAPI-приложение: онлайн-предсказание цены квартиры и метрики для Prometheus.

Запуск из директории services/:
    uvicorn ml_service.main:app --host 0.0.0.0 --port 8081
"""
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator

from ml_service.handler import FastApiHandler
from ml_service.metrics import VALIDATION_ERRORS
from ml_service.schemas import FLAT_EXAMPLE, FlatFeatures, PredictionResponse

# поля, которые могут попасть в метку метрики ошибок (метки не должны разрастаться)
KNOWN_FIELDS = set(FlatFeatures.model_fields) | {'user_id'}

handler: FastApiHandler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # модель грузим один раз при старте, а не на каждый запрос
    global handler
    handler = FastApiHandler()
    yield


app = FastAPI(
    title='Сервис оценки стоимости квартир',
    description='Онлайн-предсказание цены квартиры моделью CatBoost (Яндекс Недвижимость).',
    version='1.0.0',
    lifespan=lifespan,
)

# базовые HTTP-метрики: число запросов, коды ответов, время ответа, размеры; отдаются на /metrics
Instrumentator(excluded_handlers=['/metrics']).instrument(app).expose(app, include_in_schema=False)


@app.exception_handler(RequestValidationError)
async def count_validation_errors(request: Request, exc: RequestValidationError):
    """Считает ошибки валидации по полям и отдаёт стандартный ответ 422 с описанием ошибок."""
    for error in exc.errors():
        # loc: ('body', 'total_area') или ('query', 'user_id'); тело целиком - ('body',)
        loc = error.get('loc', ())
        if error.get('type') == 'json_invalid':
            field = 'json'
        elif error.get('type') == 'extra_forbidden':
            field = 'extra_field'
        elif len(loc) > 1 and loc[-1] in KNOWN_FIELDS:
            field = loc[-1]
        elif len(loc) > 1:
            field = 'other'
        else:
            field = 'body'
        VALIDATION_ERRORS.labels(field=field).inc()
    return await request_validation_exception_handler(request, exc)


@app.get('/', summary='Проверка, что сервис жив')
def health():
    return {
        'status': 'ok',
        'model': handler.meta.get('model_name'),
        'model_version': handler.meta.get('model_version'),
    }


@app.post('/api/price/', response_model=PredictionResponse, summary='Предсказать цену квартиры')
def get_prediction(
    user_id: str = Query(
        min_length=1,
        max_length=64,
        description='Идентификатор пользователя, который запрашивает оценку',
        openapi_examples={'user': {'summary': 'id пользователя', 'value': '123'}},
    ),
    flat: FlatFeatures = Body(
        openapi_examples={
            'flat': {
                'summary': 'Однокомнатная квартира в Москве',
                'value': FLAT_EXAMPLE,
            },
            'invalid': {
                'summary': 'Ошибка: этаж выше этажности дома',
                'description': 'Сервис ответит 422 и объяснит, что не так.',
                'value': {**FLAT_EXAMPLE, 'floor': 15, 'floors_total': 12},
            },
        },
    ),
):
    """Возвращает предсказанную цену квартиры в рублях: `{"user_id": ..., "prediction": ...}`."""
    try:
        return handler.handle(user_id, flat)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'не удалось получить предсказание: {exc}') from exc
