"""Проверки результата и взаимодействий OrderService с моками зависимостей."""

from unittest.mock import Mock, call

import pytest

from order_service import OrderService


@pytest.mark.parametrize(
    "order_id, data",
    [
        pytest.param(1, {"item": "Book", "price": 500}, id="book"),
        pytest.param(2, {"item": "Pen", "price": 50}, id="pen"),
        pytest.param(3, {"item": "Laptop", "price": 80000}, id="laptop"),
    ],
)
def test_place_order_success(order_id, data):
    storage = Mock(spec_set=["save_order", "get_order"])
    storage.save_order.return_value = True
    notifier = Mock(spec_set=["send"])
    interactions = Mock()
    interactions.attach_mock(storage, "storage")
    interactions.attach_mock(notifier, "notifier")
    service = OrderService(storage, notifier)

    result = service.place_order(order_id, data)

    assert result is True
    storage.save_order.assert_called_once_with(order_id, data)
    notifier.send.assert_called_once_with(order_id, data)
    assert interactions.mock_calls == [
        call.storage.save_order(order_id, data),
        call.notifier.send(order_id, data),
    ]


def test_place_order_save_failed():
    order_id = 4
    data = {"item": "Pencil", "price": 30}
    storage = Mock(spec_set=["save_order", "get_order"])
    storage.save_order.return_value = False
    notifier = Mock(spec_set=["send"])
    service = OrderService(storage, notifier)

    result = service.place_order(order_id, data)

    assert result is False
    storage.save_order.assert_called_once_with(order_id, data)
    notifier.send.assert_not_called()


@pytest.mark.parametrize(
    "error_type, message",
    [
        pytest.param(Exception, "Storage unavailable", id="exception"),
        pytest.param(RuntimeError, "Save failed", id="runtime-error"),
    ],
)
def test_place_order_storage_exception(error_type, message):
    order_id = 5
    data = {"item": "Notebook", "price": 100}
    error = error_type(message)
    storage = Mock(spec_set=["save_order", "get_order"])
    storage.save_order.side_effect = error
    notifier = Mock(spec_set=["send"])
    service = OrderService(storage, notifier)

    with pytest.raises(error_type) as caught:
        service.place_order(order_id, data)

    assert caught.value is error
    storage.save_order.assert_called_once_with(order_id, data)
    notifier.send.assert_not_called()


def test_get_order_existing():
    order_id = 1
    data = {"item": "Book", "price": 500}
    storage = Mock(spec_set=["save_order", "get_order"])
    storage.get_order.return_value = data
    notifier = Mock(spec_set=["send"])
    service = OrderService(storage, notifier)

    result = service.get_order(order_id)

    assert result is data
    storage.get_order.assert_called_once_with(order_id)
    storage.save_order.assert_not_called()
    notifier.send.assert_not_called()


def test_get_order_missing():
    order_id = 999
    storage = Mock(spec_set=["save_order", "get_order"])
    storage.get_order.return_value = None
    notifier = Mock(spec_set=["send"])
    service = OrderService(storage, notifier)

    result = service.get_order(order_id)

    assert result is None
    storage.get_order.assert_called_once_with(order_id)
    storage.save_order.assert_not_called()
    notifier.send.assert_not_called()
