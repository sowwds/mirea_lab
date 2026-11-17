"""Адаптер курсов: Rates By Exchange Rate API (exchangerate-api.com)."""

import math

import httpx


class RateUnavailable(Exception):
    """Внешний сервис недоступен или вернул некорректный курс."""


class ExchangeRates:
    def __init__(self, client):
        self.client = client

    def rate(self, currency):
        """Получить количество единиц валюты за один RUB."""
        try:
            response = self.client.get("https://open.er-api.com/v6/latest/RUB")
            response.raise_for_status()
            data = response.json()
            if data["result"] != "success":
                raise ValueError("Unsuccessful response")
            rate = data["rates"][currency]
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise ValueError("Invalid rate type")
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Invalid rate value")
            return rate
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            raise RateUnavailable("Exchange service unavailable") from error


def get_rates():
    """Предоставить HTTP-клиент с ограничением времени ожидания."""
    with httpx.Client(timeout=5.0) as client:
        yield ExchangeRates(client)
