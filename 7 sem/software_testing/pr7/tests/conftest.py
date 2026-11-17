"""Чистая SQLite в памяти и подменённые зависимости на каждый тест."""

from itertools import count
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database import Base, get_db, make_engine
from app.main import create_app
from app.rates import get_rates


@pytest.fixture(autouse=True)
def forbid_real_http(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Real external HTTP is forbidden in tests")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)


@pytest.fixture
def db_session():
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def rates():
    service = Mock(spec_set=["rate"])
    service.rate.side_effect = {"USD": 0.01, "EUR": 0.009}.__getitem__
    return service


@pytest.fixture
def application(db_session, rates):
    instance = create_app("sqlite://")

    def override_db():
        yield db_session

    instance.dependency_overrides[get_db] = override_db
    instance.dependency_overrides[get_rates] = lambda: rates
    yield instance
    instance.dependency_overrides.clear()


@pytest.fixture
def client(application):
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture
def admin_headers(client):
    response = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    return {"Authorization": "Bearer " + response.json()["access_token"]}


@pytest.fixture
def viewer_headers(client):
    response = client.post("/auth/login", json={"username": "viewer", "password": "viewer123"})
    assert response.status_code == 200
    return {"Authorization": "Bearer " + response.json()["access_token"]}


@pytest.fixture
def make_item(client, admin_headers):
    sequence = count(1)

    def factory(**fields):
        body = {"name": f"Ticket {next(sequence)}", "price": 100.0, "seats": 10}
        body.update(fields)
        response = client.post("/tickets", json=body, headers=admin_headers)
        assert response.status_code == 201, response.text
        return response.json()

    return factory


@pytest.fixture
def catalog(make_item):
    return [
        make_item(name="Alpha Concert", price=100),
        make_item(name="Beta Theatre", price=200),
        make_item(name="Gamma Concert", price=300),
    ]
