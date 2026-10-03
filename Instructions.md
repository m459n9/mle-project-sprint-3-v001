# Инструкции по запуску микросервиса

Каждая инструкция выполняется из корня репозитория `mle-project-sprint-3-v001`.
Если нужно перейти в поддиректорию, это указано отдельной командой.

Этапы проверяются независимо, но все они занимают порт `8081`: перед следующим этапом
остановите сервис предыдущего (команды остановки есть в конце каждого раздела).

Что нужно заранее: Python 3.10–3.13 с модулем `venv` (на Ubuntu - пакет `python3-venv`),
Docker и Docker Compose v2 (`docker compose`).

Сервис принимает признаки квартиры и возвращает предсказанную цену в рублях:

* `POST /api/price/?user_id=<id>` - тело запроса: JSON с 15 признаками квартиры;
  ответ: `{"user_id": "<id>", "prediction": <цена>}`;
* `GET /` - проверка, что сервис жив;
* `GET /docs` - Swagger с описанием полей и готовыми примерами запроса;
* `GET /metrics` - метрики для Prometheus.

## 1. FastAPI микросервис в виртуальном окружение

```bash
# команды создания виртуального окружения
# и установки необходимых библиотек в него
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r services/requirements.txt

# команда перехода в директорию
cd services

# команда запуска сервиса с помощью uvicorn
uvicorn ml_service.main:app --host 0.0.0.0 --port 8081
```

Сервис запущен, когда в логе появится `Application startup complete`.
Swagger с примером запроса: http://localhost:8081/docs.

### Пример curl-запроса к микросервису

В другом терминале:

```bash
curl -X 'POST' \
  'http://localhost:8081/api/price/?user_id=123' \
  -H 'Content-Type: application/json' \
  -d '{
    "floor": 9,
    "kitchen_area": 9.9,
    "living_area": 19.9,
    "rooms": 1,
    "is_apartment": false,
    "studio": false,
    "total_area": 35.1,
    "build_year": 1965,
    "building_type_int": 6,
    "latitude": 55.717113,
    "longitude": 37.78112,
    "ceiling_height": 2.64,
    "flats_count": 84,
    "floors_total": 12,
    "has_elevator": true
  }'
```

Ответ:

```json
{"user_id":"123","prediction":7931858.02}
```

На запрос неправильного формата сервис отвечает кодом `422` и списком ошибок по полям.
Например, если этаж квартиры больше этажности дома (`"floor": 15`, `"floors_total": 12`):

```json
{"detail":[{"type":"value_error","loc":["body","floors_total"],"msg":"Value error, этажей в доме (12) меньше, чем этаж квартиры floor (15)","input":12,"ctx":{"error":{}}}]}
```

Проверить класс-обработчик без веб-сервера (из директории `services`):

```bash
python -m ml_service.handler
```

Остановка: `Ctrl+C` в терминале с uvicorn, затем `cd ..` и `deactivate`.

## 2. FastAPI микросервис в Docker-контейнере

Запуск без Docker Compose, только командами `docker image` и `docker container`.
Переменные окружения контейнера (путь к модели, уровень логов) лежат в `services/.env`.

```bash
# команда перехода в нужную директорию
cd services

# сборка образа
docker image build . --file Dockerfile_ml_service --tag price_service:1.0

# запуск контейнера: переменные из .env, порт 8081 контейнера -> 8081 хоста
docker container run \
  --detach \
  --name price_service \
  --env-file .env \
  --publish 8081:8081 \
  price_service:1.0

# проверка: статус (healthy примерно через 30 секунд) и логи
docker container ls --filter name=price_service
docker container logs price_service
```

Если порт `8081` на хосте занят, поменяйте левую часть `--publish`, например
`--publish 8090:8081`, и обращайтесь к `http://localhost:8090`.

### Пример curl-запроса к микросервису

```bash
curl -X 'POST' \
  'http://localhost:8081/api/price/?user_id=123' \
  -H 'Content-Type: application/json' \
  -d '{"floor": 9, "kitchen_area": 9.9, "living_area": 19.9, "rooms": 1, "is_apartment": false, "studio": false, "total_area": 35.1, "build_year": 1965, "building_type_int": 6, "latitude": 55.717113, "longitude": 37.78112, "ceiling_height": 2.64, "flats_count": 84, "floors_total": 12, "has_elevator": true}'
```

Ответ: `{"user_id":"123","prediction":7931858.02}`.

Остановка и удаление контейнера:

```bash
docker container stop price_service
docker container rm price_service
cd ..
```

## 3. Docker compose для микросервиса и системы моониторинга

Поднимаются три сервиса: `ml-service` (FastAPI с моделью), `prometheus` и `grafana`.
Порты на хосте и логин/пароль Grafana задаются в `services/.env`.
Источник данных Prometheus подключается в Grafana автоматически
(`services/grafana/provisioning/datasources/prometheus.yml`).

```bash
# команда перехода в нужную директорию
cd services

# команда для запуска микросервиса в режиме docker compose
docker compose up --build --detach

# проверка: все три сервиса в статусе Up, ml-service - healthy
docker compose ps
```

Адреса сервисов (порты по умолчанию из `services/.env`):
- микросервис: http://localhost:8081 (Swagger - http://localhost:8081/docs, метрики - http://localhost:8081/metrics)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (логин и пароль - `GRAFANA_USER` / `GRAFANA_PASS` из `services/.env`)

После перезапуска виртуальной машины контейнеры лучше пересоздать:

```bash
docker compose down
docker compose up --build --detach
```

### Пример curl-запроса к микросервису

```bash
curl -X 'POST' \
  'http://localhost:8081/api/price/?user_id=123' \
  -H 'Content-Type: application/json' \
  -d '{"floor": 9, "kitchen_area": 9.9, "living_area": 19.9, "rooms": 1, "is_apartment": false, "studio": false, "total_area": 35.1, "build_year": 1965, "building_type_int": 6, "latitude": 55.717113, "longitude": 37.78112, "ceiling_height": 2.64, "flats_count": 84, "floors_total": 12, "has_elevator": true}'
```

Ответ: `{"user_id":"123","prediction":7931858.02}`.

Метрики экспортёра доступны на http://localhost:8081/metrics. В Prometheus
(http://localhost:9090) на странице **Status → Target health** цель `ml-service` должна
быть в состоянии `UP`, а в **Query** находятся базовые метрики экспортёра, например
`http_requests_total`, `http_request_duration_seconds_bucket`.

Остановка (контейнеры удаляются вместе с накопленными метриками и импортированным
дашбордом, его можно заново загрузить из `dashboard.json`):

```bash
docker compose down
cd ..
```

## 4. Скрипт симуляции нагрузки

Скрипт `scripts/simulate_load.py` по умолчанию в течение 300 секунд шлёт на сервис
около 5 запросов в секунду (частота плавно меняется от 2 до 8 запросов/с), всего около
1 500 запросов. Среди них:

* ~10% заведомо ошибочных (пропущено поле, этаж выше этажности, строка вместо числа,
  лишнее поле, пустой `user_id`) - сервис отвечает `422`;
* корректные квартиры вне диапазона обучающей выборки (Санкт-Петербург, площадь больше
  120 м², потолки выше 2.88 м): в первой половине прогона их ~3%, во второй ~20% - так
  имитируется дрейф данных;
* остальное - обычные московские квартиры, похожие на обучающую выборку.

Каждые 10 секунд скрипт печатает, сколько запросов отправлено и с какими кодами ответа.

Сервисы этапа 3 должны быть запущены. Скрипт использует `requests` из того же
виртуального окружения, что и этап 1:

```bash
# если окружение ещё не создано - см. этап 1
source .venv/bin/activate

# нагрузка по умолчанию: ~5 запросов/с в течение 300 секунд
python scripts/simulate_load.py

# параметры можно менять: длительность, частота, доли ошибок и дрейфа, адрес сервиса
python scripts/simulate_load.py --duration 600 --rps 10 --error-share 0.1 --drift-share 0.1 --url http://localhost:8081
```

### Загрузка дашборда в Grafana

1. Откройте Grafana http://localhost:3000 и войдите с логином и паролем
   из `services/.env` (`GRAFANA_USER` / `GRAFANA_PASS`).
2. Источник данных `prometheus` уже подключён автоматически, и в `dashboard.json`
   прописан его uid. Если вы подключали источник вручную, синхронизируйте uid скриптом
   из шаблона (из корня репозитория, при активированном `.venv`):

   ```bash
   python fix_datasource_uid.py
   ```

3. **Dashboards → New → Import → Upload dashboard JSON file** → выберите
   `dashboard.json` → **Import**.
4. Запустите скрипт нагрузки: панели обновляются каждые 10 секунд, по умолчанию
   показаны последние 15 минут.

Адреса сервисов:
- микросервис: http://localhost:8081 (Swagger - http://localhost:8081/docs, метрики - http://localhost:8081/metrics)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Порты задаются в `services/.env` (`APP_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT`).
При работе на виртуальной машине пробросьте эти порты на свой компьютер,
например через вкладку **Ports** в VS Code.
