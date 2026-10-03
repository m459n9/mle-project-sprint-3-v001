"""Симуляция нагрузки на сервис оценки стоимости квартир.

Скрипт в течение --duration секунд шлёт POST-запросы на /api/price/ с частотой
около --rps запросов в секунду (частота плавно «дышит», чтобы на графиках была
видна динамика). Запросы трёх видов:

* обычные квартиры, похожие на обучающую выборку (Москва, 1-5 комнат);
* корректные, но непривычные для модели квартиры (другой город, большая площадь,
  высокие потолки) - их доля растёт ко второй половине прогона и имитирует
  дрейф данных, это видно на ML-панелях дашборда;
* заведомо ошибочные запросы (пропущено поле, этаж выше этажности, строка
  вместо числа и т.п.) - сервис отвечает 422, это видно на панелях ошибок.

Запуск из корня репозитория (сервисы этапа 3 уже подняты):
    python scripts/simulate_load.py
    python scripts/simulate_load.py --duration 600 --rps 10
"""
import argparse
import math
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

# доли типов домов и комнат примерно как в clean_flats_dataset
BUILDING_TYPES = [0, 1, 2, 3, 4, 6]
BUILDING_WEIGHTS = [0.01, 0.08, 0.065, 0.002, 0.76, 0.083]
ROOMS = [1, 2, 3, 4, 5]
ROOMS_WEIGHTS = [0.36, 0.40, 0.224, 0.015, 0.001]
AREA_BY_ROOMS = {1: (34, 4.5), 2: (50, 7), 3: (66, 9), 4: (84, 10), 5: (100, 10)}


def normal_flat(rng: random.Random) -> dict:
    """Квартира, похожая на обучающую выборку."""
    rooms = rng.choices(ROOMS, ROOMS_WEIGHTS)[0]
    mean, std = AREA_BY_ROOMS[rooms]
    total_area = round(min(max(rng.gauss(mean, std), 20), 115), 1)
    kitchen_area = round(min(max(rng.gauss(8.5, 2), 5), 14), 1)
    living_area = round(min(total_area * rng.uniform(0.45, 0.65), total_area - kitchen_area), 1)
    floors_total = rng.choice([5, 9, 9, 12, 14, 16, 17, 17, 22, 25])
    return {
        'floor': rng.randint(1, floors_total),
        'kitchen_area': kitchen_area,
        'living_area': living_area,
        'rooms': rooms,
        'is_apartment': rng.random() < 0.003,
        'studio': False,
        'total_area': total_area,
        'build_year': int(min(max(rng.gauss(1983, 17), 1930), 2022)),
        'building_type_int': rng.choices(BUILDING_TYPES, BUILDING_WEIGHTS)[0],
        # Москва: центр и разброс координат как в обучающей выборке
        'latitude': round(min(max(rng.gauss(55.72, 0.115), 55.45), 56.0), 6),
        'longitude': round(min(max(rng.gauss(37.59, 0.165), 37.2), 37.94), 6),
        'ceiling_height': rng.choice([2.5, 2.6, 2.64, 2.64, 2.7, 2.75, 2.8]),
        'flats_count': floors_total * rng.randint(4, 24),
        'floors_total': floors_total,
        'has_elevator': floors_total > 5 or rng.random() < 0.2,
    }


def unusual_flat(rng: random.Random) -> dict:
    """Корректная квартира вне диапазона обучающей выборки: модель экстраполирует."""
    flat = normal_flat(rng)
    kind = rng.choice(['spb', 'big', 'loft'])
    if kind == 'spb':
        # Санкт-Петербург: модель обучена только на Москве
        flat.update(latitude=round(rng.uniform(59.85, 60.05), 6), longitude=round(rng.uniform(30.2, 30.45), 6))
    elif kind == 'big':
        # большая квартира: в train общая площадь не больше 120 м²
        flat.update(rooms=rng.randint(4, 6), total_area=round(rng.uniform(130, 250), 1),
                    living_area=round(rng.uniform(80, 120), 1), kitchen_area=round(rng.uniform(16, 30), 1))
    else:
        # новостройка с высокими потолками: в train потолки до 2.88 м
        flat.update(ceiling_height=round(rng.uniform(3.0, 3.6), 2), build_year=2024)
    return flat


def invalid_request(rng: random.Random) -> tuple[dict, str]:
    """Запрос, который сервис должен отклонить с кодом 422. Возвращает (тело, user_id)."""
    flat = normal_flat(rng)
    user_id = str(rng.randint(1, 10_000))
    kind = rng.choice(['missing', 'floor', 'area', 'type', 'extra', 'negative', 'user_id'])
    if kind == 'missing':
        flat.pop(rng.choice(list(flat)))
    elif kind == 'floor':
        flat['floor'] = flat['floors_total'] + rng.randint(1, 5)
    elif kind == 'area':
        flat['living_area'] = flat['total_area'] + 10
    elif kind == 'type':
        flat['rooms'] = 'три'
    elif kind == 'extra':
        flat['price'] = 10_000_000
    elif kind == 'negative':
        flat['total_area'] = -flat['total_area']
    else:
        user_id = ''
    return flat, user_id


class LoadGenerator:
    def __init__(self, url: str, error_share: float, drift_share: float, seed: int):
        self.url = url.rstrip('/') + '/api/price/'
        self.error_share = error_share
        self.drift_share = drift_share
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.stats = Counter()
        self.local = threading.local()

    def session(self) -> requests.Session:
        # у каждого потока своя сессия: переиспользуем соединения
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
        return self.local.session

    def make_request(self, progress: float) -> tuple[dict, str]:
        with self.lock:
            rng = random.Random(self.rng.random())
        # во второй половине прогона доля непривычных квартир растёт - имитация дрейфа
        drift_share = self.drift_share * (0.3 if progress < 0.5 else 2.0)
        roll = rng.random()
        if roll < self.error_share:
            return invalid_request(rng)
        if roll < self.error_share + drift_share:
            return unusual_flat(rng), str(rng.randint(1, 10_000))
        return normal_flat(rng), str(rng.randint(1, 10_000))

    def send(self, progress: float) -> None:
        body, user_id = self.make_request(progress)
        try:
            response = self.session().post(self.url, params={'user_id': user_id}, json=body, timeout=10)
            key = str(response.status_code)
        except requests.RequestException as exc:
            key = type(exc).__name__
        with self.lock:
            self.stats[key] += 1

    def run(self, duration: float, rps: float, workers: int) -> None:
        start = time.monotonic()
        next_report = start + 10
        sent = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while (elapsed := time.monotonic() - start) < duration:
                progress = elapsed / duration
                # частота «дышит» с периодом 2 минуты: от 0.4 до 1.6 от заданной
                current_rps = rps * (1 + 0.6 * math.sin(2 * math.pi * elapsed / 120))
                pool.submit(self.send, progress)
                sent += 1
                time.sleep(1 / max(current_rps, 0.1))
                if time.monotonic() >= next_report:
                    with self.lock:
                        summary = dict(sorted(self.stats.items()))
                    print(f'[{elapsed:5.0f} c] отправлено {sent}, ответы: {summary}', flush=True)
                    next_report += 10
        print(f'Готово: {sent} запросов за {time.monotonic() - start:.0f} c, ответы: {dict(sorted(self.stats.items()))}')


def main():
    parser = argparse.ArgumentParser(description='Нагрузка на сервис оценки стоимости квартир')
    parser.add_argument('--url', default='http://localhost:8081', help='адрес сервиса (по умолчанию %(default)s)')
    parser.add_argument('--duration', type=float, default=300, help='длительность, секунд (%(default)s)')
    parser.add_argument('--rps', type=float, default=5, help='средняя частота запросов в секунду (%(default)s)')
    parser.add_argument('--error-share', type=float, default=0.1, help='доля ошибочных запросов (%(default)s)')
    parser.add_argument('--drift-share', type=float, default=0.1,
                        help='базовая доля непривычных квартир, во второй половине x2 (%(default)s)')
    parser.add_argument('--workers', type=int, default=8, help='параллельных потоков (%(default)s)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f'Нагрузка на {args.url}: ~{args.rps} запросов/с в течение {args.duration:.0f} c')
    generator = LoadGenerator(args.url, args.error_share, args.drift_share, args.seed)
    generator.run(args.duration, args.rps, args.workers)


if __name__ == '__main__':
    main()
