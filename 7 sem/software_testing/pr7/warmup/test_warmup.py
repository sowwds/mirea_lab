"""Один регрессионный сценарий для исходного и исправленного примеров."""

import os
from importlib import import_module

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def warmup_client():
    module = import_module("warmup." + os.getenv("WARMUP_APP", "app_fixed"))
    module.ITEMS.clear()
    if hasattr(module, "NEXT_ID"):
        module.NEXT_ID = 1
    with TestClient(module.app) as client:
        yield client
    module.ITEMS.clear()


def test_create_after_delete_preserves_existing_item(warmup_client):
    first = warmup_client.post("/items", json={"name": "A", "price": 10})
    second = warmup_client.post("/items", json={"name": "B", "price": 20})
    assert first.status_code == second.status_code == 201
    assert warmup_client.delete(f"/items/{first.json()['id']}").status_code == 204

    third = warmup_client.post("/items", json={"name": "C", "price": 30})

    assert third.status_code == 201
    assert third.json()["id"] != second.json()["id"]
    module = import_module("warmup." + os.getenv("WARMUP_APP", "app_fixed"))
    assert module.ITEMS[second.json()["id"]]["name"] == "B"
