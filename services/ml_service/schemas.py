"""Схемы запроса и ответа: по ним FastAPI валидирует вход и строит Swagger."""
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

# пример квартиры: его видно в Swagger на /docs, он же в Instructions.md
FLAT_EXAMPLE = {
    'floor': 9,
    'kitchen_area': 9.9,
    'living_area': 19.9,
    'rooms': 1,
    'is_apartment': False,
    'studio': False,
    'total_area': 35.1,
    'build_year': 1965,
    'building_type_int': 6,
    'latitude': 55.717113,
    'longitude': 37.781120,
    'ceiling_height': 2.64,
    'flats_count': 84,
    'floors_total': 12,
    'has_elevator': True,
}


class FlatFeatures(BaseModel):
    """Признаки квартиры и дома - вход модели.

    Границы полей - физический смысл (площадь > 0, широта от -90 до 90), а не
    диапазон обучающей выборки: квартира вне опыта модели всё равно получит
    оценку, а сервис отметит это в метрике price_model_out_of_range_total.
    """

    # лишние поля (например, price) - ошибка, а не молчаливый пропуск
    model_config = ConfigDict(extra='forbid', json_schema_extra={'examples': [FLAT_EXAMPLE]})

    floor: int = Field(ge=1, le=200, description='Этаж квартиры')
    kitchen_area: float = Field(ge=0, le=500, description='Площадь кухни, м²')
    living_area: float = Field(ge=0, le=1000, description='Жилая площадь, м² (0 - неизвестна)')
    rooms: int = Field(ge=1, le=20, description='Число комнат')
    is_apartment: bool = Field(description='Апартаменты')
    studio: bool = Field(description='Студия')
    total_area: float = Field(gt=0, le=1000, description='Общая площадь, м²')
    build_year: int = Field(ge=1800, le=2035, description='Год постройки дома')
    building_type_int: int = Field(ge=0, le=6, description='Код типа дома, 0-6')
    latitude: float = Field(ge=-90, le=90, description='Широта')
    longitude: float = Field(ge=-180, le=180, description='Долгота')
    ceiling_height: float = Field(gt=1.5, le=10, description='Высота потолков, м')
    flats_count: int = Field(ge=1, le=10000, description='Квартир в доме')
    floors_total: int = Field(ge=1, le=200, description='Этажей в доме')
    has_elevator: bool = Field(description='Есть лифт')

    # проверки, где участвуют несколько полей: валидатор вешаем на поле, которое
    # объявлено позже (к этому моменту остальные уже проверены и лежат в info.data),
    # чтобы в ответе 422 ошибка указывала на конкретное поле

    @field_validator('total_area')
    @classmethod
    def check_areas(cls, total_area, info: ValidationInfo):
        for part in ('living_area', 'kitchen_area'):
            value = info.data.get(part)
            if value is not None and value > total_area:
                raise ValueError(f'общая площадь ({total_area}) меньше, чем {part} ({value})')
        return total_area

    @field_validator('floors_total')
    @classmethod
    def check_floor(cls, floors_total, info: ValidationInfo):
        floor = info.data.get('floor')
        if floor is not None and floor > floors_total:
            raise ValueError(f'этажей в доме ({floors_total}) меньше, чем этаж квартиры floor ({floor})')
        return floors_total


class PredictionResponse(BaseModel):
    """Ответ сервиса: id пользователя и предсказанная цена, руб."""

    model_config = ConfigDict(json_schema_extra={'examples': [{'user_id': '123', 'prediction': 7931858.02}]})

    user_id: str
    prediction: float
