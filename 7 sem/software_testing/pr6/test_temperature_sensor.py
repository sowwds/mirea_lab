"""Проверки событий, подписок и fixtures датчика температуры."""

import logging
from unittest.mock import Mock, call

import pytest


def test_initial_temperature(sensor):
    assert sensor.temperature == 0.0


def test_temperature_change(ready_sensor):
    callback = Mock()
    ready_sensor.subscribe(callback)

    ready_sensor.set_temperature(25)

    assert ready_sensor.temperature == 25
    callback.assert_called_once_with(25)


@pytest.mark.parametrize("temperature", [20, 20.0], ids=["integer", "float"])
def test_same_temperature(ready_sensor, temperature):
    callback = Mock()
    ready_sensor.subscribe(callback)

    ready_sensor.set_temperature(temperature)

    assert ready_sensor.temperature == 20
    callback.assert_not_called()


@pytest.mark.parametrize("subscriber_count", [2, 3], ids=["two", "three"])
def test_multiple_subscribers(ready_sensor, subscriber_count):
    callbacks = [Mock() for _ in range(subscriber_count)]
    for callback in callbacks:
        ready_sensor.subscribe(callback)

    ready_sensor.set_temperature(30)

    assert ready_sensor.temperature == 30
    for callback in callbacks:
        callback.assert_called_once_with(30)


def test_unsubscribe(ready_sensor):
    removed = Mock()
    remaining = Mock()
    ready_sensor.subscribe(removed)
    ready_sensor.subscribe(remaining)
    ready_sensor.unsubscribe(removed)

    ready_sensor.set_temperature(25)

    removed.assert_not_called()
    remaining.assert_called_once_with(25)


def test_unsubscribe_unknown(ready_sensor):
    subscribed = Mock()
    unknown = Mock()
    ready_sensor.subscribe(subscribed)
    ready_sensor.unsubscribe(unknown)

    ready_sensor.set_temperature(25)

    unknown.assert_not_called()
    subscribed.assert_called_once_with(25)


def test_repeated_unsubscribe(ready_sensor):
    removed = Mock()
    remaining = Mock()
    ready_sensor.subscribe(removed)
    ready_sensor.subscribe(remaining)
    ready_sensor.unsubscribe(removed)
    ready_sensor.unsubscribe(removed)

    ready_sensor.set_temperature(25)

    removed.assert_not_called()
    remaining.assert_called_once_with(25)


@pytest.mark.parametrize(
    "temperature",
    [
        pytest.param(-10, id="negative"),
        pytest.param(0, id="zero"),
        pytest.param(20.5, id="fractional"),
        pytest.param(30, id="positive"),
        pytest.param(10**6, id="large"),
    ],
)
def test_temperature_values(ready_sensor, temperature):
    callback = Mock()
    ready_sensor.subscribe(callback)

    ready_sensor.set_temperature(temperature)

    assert ready_sensor.temperature == temperature
    callback.assert_called_once_with(temperature)


def test_temperature_sequence(ready_sensor):
    callback = Mock()
    ready_sensor.subscribe(callback)

    for temperature in [25, 25, -5, -5, 20]:
        ready_sensor.set_temperature(temperature)

    assert ready_sensor.temperature == 20
    assert callback.call_args_list == [call(25), call(-5), call(20)]


@pytest.mark.parametrize("failure_position", [0, 1, 2], ids=["first", "middle", "last"])
def test_subscriber_exception(ready_sensor, caplog, failure_position):
    first_healthy = Mock()
    second_healthy = Mock()
    error = RuntimeError("Subscriber failed")
    failing = Mock(side_effect=error)
    callbacks = [first_healthy, second_healthy]
    callbacks.insert(failure_position, failing)
    for callback in callbacks:
        ready_sensor.subscribe(callback)

    with caplog.at_level(logging.ERROR, logger="temperature_sensor"):
        ready_sensor.set_temperature(25)

    assert ready_sensor.temperature == 25
    failing.assert_called_once_with(25)
    first_healthy.assert_called_once_with(25)
    second_healthy.assert_called_once_with(25)
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage() == "Ошибка подписчика при изменении температуры"
    assert caplog.records[0].exc_info[1] is error


def test_duplicate_subscription(ready_sensor):
    callback = Mock()
    ready_sensor.subscribe(callback)
    ready_sensor.subscribe(callback)

    ready_sensor.set_temperature(25)

    callback.assert_called_once_with(25)
