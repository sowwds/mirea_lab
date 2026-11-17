"""Событийная модель датчика температуры для практической работы № 6."""

import logging
from collections.abc import Callable


logger = logging.getLogger(__name__)


class TemperatureSensor:
    """Хранить температуру и уведомлять подписчиков об изменениях."""

    def __init__(self, initial_temperature: float = 0.0):
        """Задать начальную температуру и пустой список подписчиков."""
        self._temperature = initial_temperature
        self._subscribers: list[Callable[[float], None]] = []

    @property
    def temperature(self) -> float:
        """Вернуть текущую температуру."""
        return self._temperature

    def subscribe(self, callback: Callable[[float], None]) -> None:
        """Добавить подписчика, если он ещё не подписан."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[float], None]) -> None:
        """Удалить подписчика; отсутствие подписки не считать ошибкой."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def set_temperature(self, temperature: float) -> None:
        """Обновить температуру и уведомить только при изменении значения."""
        if temperature == self._temperature:
            return
        self._temperature = temperature
        self._notify(temperature)

    def _notify(self, temperature: float) -> None:
        """Вызвать подписчиков, изолируя ошибки отдельных обработчиков."""
        for callback in tuple(self._subscribers):
            try:
                callback(temperature)
            except Exception:
                logger.exception("Ошибка подписчика при изменении температуры")
