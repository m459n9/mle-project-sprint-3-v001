# Релиз модели оценки стоимости квартир в продакшен

Проект третьего спринта. Модель оценки цены квартиры для Яндекс Недвижимости, полученная
во втором спринте (генерация и отбор признаков, подбор гиперпараметров), выводится в
продакшен:

1. **FastAPI-микросервис** - коллеги получают предсказание онлайн HTTP-запросом;
   класс-обработчик валидирует вход и отвечает `{"user_id": ..., "prediction": ...}`.
2. **Docker** - сервис упакован в образ; порты и настройки вынесены в `services/.env`,
   поэтому его можно развернуть на любой виртуальной машине рядом с другими сервисами.
3. **Prometheus и Grafana** - мониторинг сервиса, данных и модели в режиме Docker Compose.
4. **Дашборд в Grafana** - прикладные, ML- и инфраструктурные метрики, плюс скрипт
   симуляции нагрузки.

Имя бакета: `s3-student-mle-20260908-7056ef1b44`

Как запускать каждый этап - в [Instructions.md](Instructions.md), какие метрики выбраны
и почему дашборд устроен именно так - в [Monitoring.md](Monitoring.md).

## Модель

Финальная модель второго спринта (версия 4 `flats_price_model` в MLflow): sklearn-пайплайн
«ручные признаки → `ColumnTransformer` (one-hot, полиномы, дискретизация, масштабирование)
→ 11 признаков после forward selection → `CatBoostRegressor`» с гиперпараметрами Optuna.

Модель лежит в `services/models/price_model.pkl`, рядом `model_meta.json` - метрики,
гиперпараметры и диапазоны признаков в обучающей выборке. Сервис сверяет с этими
диапазонами каждый запрос, чтобы замечать дрейф данных.

Пайплайн переобучен скриптом `scripts/train_model.py` на тех же данных
(`clean_flats_dataset`), с тем же разбиением train/test и теми же гиперпараметрами, но на
актуальных версиях библиотек (scikit-learn 1.7, CatBoost 1.2.10). Это нужно, чтобы модель
загружалась на Python 3.10–3.13 без конфликта версий.

| | MAE, ₽ | RMSE, ₽ | MAPE | R² |
|---|---|---|---|---|
| Спринт 2, версия 4 в MLflow | 1 639 116 | 2 029 011 | 15.99% | 0.659 |
| Модель в сервисе | 1 641 098 | 2 031 685 | 15.93% | 0.659 |

Разница 0.1% по MAE - из-за версии CatBoost.

## Стек

Python 3.11, FastAPI, Uvicorn, Pydantic, scikit-learn, CatBoost, pandas, Docker,
Docker Compose, Prometheus, Grafana, `prometheus_client`,
`prometheus_fastapi_instrumentator`, requests.

## Структура репозитория

```
├── README.md
├── Instructions.md              инструкции по запуску всех этапов
├── Monitoring.md                выбранные метрики и описание дашборда
├── dashboard.json               дашборд Grafana
├── dashboard.jpg                скриншот дашборда
├── fix_datasource_uid.py        подстановка uid источника данных в dashboard.json (из шаблона)
├── scripts/
│   ├── simulate_load.py         симуляция нагрузки на сервис
│   └── train_model.py           воспроизведение финальной модели второго спринта
└── services/
    ├── .env                     переменные окружения: порты, путь к модели, доступ в Grafana
    ├── requirements.txt         зависимости сервиса и скриптов
    ├── Dockerfile_ml_service    образ FastAPI-микросервиса
    ├── docker-compose.yaml      микросервис + Prometheus + Grafana
    ├── ml_service/
    │   ├── main.py              FastAPI-приложение, эндпоинты, экспорт метрик
    │   ├── handler.py           FastApiHandler: загрузка модели, валидация, предсказание
    │   ├── schemas.py           схемы запроса и ответа (Pydantic), пример для Swagger
    │   ├── metrics.py           собственные метрики prometheus_client
    │   └── features.py          шаги sklearn-пайплайна модели
    ├── models/
    │   ├── price_model.pkl      модель
    │   └── model_meta.json      метаданные модели
    ├── prometheus/
    │   └── prometheus.yml       конфиг Prometheus
    └── grafana/provisioning/datasources/
        └── prometheus.yml       автоподключение Prometheus в Grafana
```
