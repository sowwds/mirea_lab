"""Арифметические операции для практической работы № 4."""


class Calculator:
    """Калькулятор для чисел int и float."""

    def add(self, a: int | float, b: int | float) -> int | float:
        """Вернуть сумму двух чисел."""
        return a + b

    def subtract(self, a: int | float, b: int | float) -> int | float:
        """Вернуть разность двух чисел."""
        return a - b

    def multiply(self, a: int | float, b: int | float) -> int | float:
        """Вернуть произведение двух чисел."""
        return a * b

    def divide(self, a: int | float, b: int | float) -> float:
        """Вернуть частное; при нулевом делителе вызвать ValueError."""
        if b == 0:
            raise ValueError("Деление на ноль")
        return a / b
