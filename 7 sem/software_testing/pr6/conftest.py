"""Общее окружение тестов TemperatureSensor."""

import pytest

from temperature_sensor import TemperatureSensor


@pytest.fixture
def sensor():
    """Создать новый сенсор с начальной температурой 0.0."""
    return TemperatureSensor()


@pytest.fixture
def ready_sensor(sensor):
    """Подготовить отдельный для каждого случая сенсор с температурой 20."""
    sensor.set_temperature(20)
    return sensor
