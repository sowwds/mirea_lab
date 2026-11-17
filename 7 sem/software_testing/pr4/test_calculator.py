"""Независимые юнит-тесты Calculator по схеме Arrange — Act — Assert."""

import pytest

from calculator import Calculator


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param(2, 3, 5, id="positive"),
        pytest.param(0, 0, 0, id="zeros"),
        pytest.param(7, 0, 7, id="zero-operand"),
        pytest.param(-2, -3, -5, id="negative"),
        pytest.param(-5, 5, 0, id="opposite"),
        pytest.param(10**18, 10**18, 2 * 10**18, id="large"),
    ],
)
def test_add_integers(a, b, expected):
    calculator = Calculator()

    result = calculator.add(a, b)

    assert result == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param(0.1, 0.2, 0.3, id="fractional"),
        pytest.param(2, 0.5, 2.5, id="mixed-types"),
    ],
)
def test_add_floats(a, b, expected):
    calculator = Calculator()

    result = calculator.add(a, b)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param(7, 3, 4, id="positive"),
        pytest.param(0, 0, 0, id="zeros"),
        pytest.param(0, 5, -5, id="negative-result"),
        pytest.param(-7, -3, -4, id="negative-operands"),
        pytest.param(10**18, -(10**18), 2 * 10**18, id="large"),
    ],
)
def test_subtract_integers(a, b, expected):
    calculator = Calculator()

    result = calculator.subtract(a, b)

    assert result == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param(0.3, 0.1, 0.2, id="fractional"),
        pytest.param(2, 0.5, 1.5, id="mixed-types"),
    ],
)
def test_subtract_floats(a, b, expected):
    calculator = Calculator()

    result = calculator.subtract(a, b)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param(6, 7, 42, id="positive"),
        pytest.param(0, 5, 0, id="zero-left"),
        pytest.param(5, 0, 0, id="zero-right"),
        pytest.param(-3, 4, -12, id="one-negative"),
        pytest.param(-3, -4, 12, id="two-negative"),
        pytest.param(10**18, 10**18, 10**36, id="large"),
    ],
)
def test_multiply_integers(a, b, expected):
    calculator = Calculator()

    result = calculator.multiply(a, b)

    assert result == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param(0.1, 0.2, 0.02, id="fractional"),
        pytest.param(3, 0.5, 1.5, id="mixed-types"),
    ],
)
def test_multiply_floats(a, b, expected):
    calculator = Calculator()

    result = calculator.multiply(a, b)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param(10, 2, 5.0, id="positive"),
        pytest.param(0, 5, 0.0, id="zero-numerator"),
        pytest.param(-10, 2, -5.0, id="negative-numerator"),
        pytest.param(10, -2, -5.0, id="negative-denominator"),
        pytest.param(-10, -2, 5.0, id="two-negative"),
        pytest.param(10**18, 10**9, 10**9, id="large"),
        pytest.param(1, 3, 0.3333333333333333, id="repeating-fraction"),
        pytest.param(7.5, 2.5, 3.0, id="float-operands"),
        pytest.param(1, 0.5, 2.0, id="mixed-types"),
    ],
)
def test_divide(a, b, expected):
    calculator = Calculator()

    result = calculator.divide(a, b)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "a, b",
    [
        pytest.param(10, 0, id="integer-zero"),
        pytest.param(0, 0, id="zero-by-zero"),
        pytest.param(1.5, 0.0, id="float-zero"),
        pytest.param(-10, -0.0, id="negative-zero"),
    ],
)
def test_divide_by_zero(a, b):
    calculator = Calculator()

    with pytest.raises(ValueError, match="^Деление на ноль$"):
        calculator.divide(a, b)
