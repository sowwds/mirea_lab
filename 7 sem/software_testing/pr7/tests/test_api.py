"""API-проверки HTTP-контракта, ролей, SQLite и подменённых курсов."""

import pytest

from app.rates import RateUnavailable


def test_health_returns_ok_without_token(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_login_returns_bearer_token(client):
    response = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    assert set(response.json()) == {"access_token", "token_type"}
    assert isinstance(response.json()["access_token"], str)
    assert response.json()["access_token"]
    assert response.json()["token_type"] == "bearer"


@pytest.mark.parametrize("username,password", [
    pytest.param("admin", "wrong", id="wrong-password"),
    pytest.param("unknown", "admin123", id="unknown-user"),
])
def test_bad_login_returns_401(client, username, password):
    response = client.post("/auth/login", json={"username": username, "password": password})
    assert response.status_code == 401


@pytest.mark.parametrize("body", [
    pytest.param({}, id="missing-both"),
    pytest.param({"username": "admin"}, id="missing-password"),
    pytest.param({"username": "", "password": "x"}, id="empty-username"),
    pytest.param({"username": "admin", "password": 12}, id="wrong-password-type"),
])
def test_invalid_login_returns_422(client, body):
    assert client.post("/auth/login", json=body).status_code == 422


def test_without_token_returns_401(client):
    response = client.get("/tickets")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("authorization", ["Bearer broken", "Basic broken"], ids=["unknown", "scheme"])
def test_invalid_token_returns_401(client, authorization):
    assert client.get("/tickets", headers={"Authorization": authorization}).status_code == 401


@pytest.mark.parametrize("method,suffix,body", [
    pytest.param("POST", "", {"name": "New", "price": 1}, id="post"),
    pytest.param("PUT", "/{id}", {"name": "New", "price": 1, "seats": 0}, id="put"),
    pytest.param("PATCH", "/{id}", {"price": 1}, id="patch"),
    pytest.param("DELETE", "/{id}", None, id="delete"),
    pytest.param("POST", "/{id}/seats", {"delta": 1}, id="seats"),
])
def test_viewer_cannot_write(client, viewer_headers, make_item, method, suffix, body):
    ticket = make_item()
    path = "/tickets" + suffix.format(id=ticket["id"])
    response = client.request(method, path, json=body, headers=viewer_headers)
    assert response.status_code == 403
    assert client.get(f"/tickets/{ticket['id']}", headers=viewer_headers).json() == ticket
    assert len(client.get("/tickets", headers=viewer_headers).json()) == 1


def test_create_ticket_returns_complete_object(client, admin_headers):
    response = client.post("/tickets", json={"name": "Concert", "price": 125.5}, headers=admin_headers)
    assert response.status_code == 201
    assert response.json() == {"id": 1, "name": "Concert", "price": 125.5, "seats": 0}


def test_get_ticket_as_viewer(client, viewer_headers, make_item):
    ticket = make_item()
    response = client.get(f"/tickets/{ticket['id']}", headers=viewer_headers)
    assert response.status_code == 200
    assert response.json() == ticket


def test_empty_list_returns_empty_array(client, viewer_headers):
    response = client.get("/tickets", headers=viewer_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_list_sorted_by_id(client, viewer_headers, catalog):
    response = client.get("/tickets", headers=viewer_headers)
    assert response.status_code == 200
    assert response.json() == catalog


def test_put_replaces_all_fields(client, admin_headers, make_item):
    ticket = make_item()
    body = {"name": "Replacement", "price": 250, "seats": 30}
    response = client.put(f"/tickets/{ticket['id']}", json=body, headers=admin_headers)
    assert response.status_code == 200
    expected = {"id": ticket["id"], **body}
    assert response.json() == expected
    assert client.get(f"/tickets/{ticket['id']}", headers=admin_headers).json() == expected


def test_patch_preserves_unspecified_fields(client, admin_headers, make_item):
    ticket = make_item()
    response = client.patch(f"/tickets/{ticket['id']}", json={"price": 50}, headers=admin_headers)
    assert response.status_code == 200
    assert response.json() == {**ticket, "price": 50}


def test_delete_returns_204_then_get_404(client, admin_headers, make_item):
    ticket = make_item()
    response = client.delete(f"/tickets/{ticket['id']}", headers=admin_headers)
    assert response.status_code == 204
    assert response.content == b""
    assert client.get(f"/tickets/{ticket['id']}", headers=admin_headers).status_code == 404


@pytest.mark.parametrize("method,suffix,body", [
    pytest.param("GET", "", None, id="get"),
    pytest.param("PUT", "", {"name": "X", "price": 1, "seats": 0}, id="put"),
    pytest.param("PATCH", "", {"price": 1}, id="patch"),
    pytest.param("DELETE", "", None, id="delete"),
    pytest.param("POST", "/seats", {"delta": 1}, id="seats"),
    pytest.param("GET", "/price?currency=USD", None, id="price"),
])
def test_missing_ticket_returns_404(client, admin_headers, method, suffix, body):
    response = client.request(method, "/tickets/999" + suffix, json=body, headers=admin_headers)
    assert response.status_code == 404


@pytest.mark.parametrize("body", [
    pytest.param({"name": "", "price": 1}, id="empty-name"),
    pytest.param({"name": "   ", "price": 1}, id="spaces-only"),
    pytest.param({"name": "x" * 101, "price": 1}, id="long-name"),
    pytest.param({"name": "X", "price": 0}, id="zero-price"),
    pytest.param({"name": "X", "price": -1}, id="negative-price"),
    pytest.param({"name": "X", "price": "10"}, id="string-price"),
    pytest.param({"name": "X", "price": 1, "seats": -1}, id="negative-seats"),
    pytest.param({"name": "X", "price": 1, "seats": 301}, id="seats-over-limit"),
    pytest.param({"name": "X", "price": 1, "seats": 1.5}, id="fractional-seats"),
    pytest.param({"name": "X", "price": 1, "seats": True}, id="boolean-seats"),
    pytest.param({"name": "X", "price": 1, "extra": 2}, id="extra-field"),
    pytest.param({"name": 12, "price": 1}, id="numeric-name"),
    pytest.param({"price": 1}, id="missing-name"),
])
def test_invalid_create_returns_422_without_writing(client, admin_headers, body):
    assert client.post("/tickets", json=body, headers=admin_headers).status_code == 422
    assert client.get("/tickets", headers=admin_headers).json() == []


@pytest.mark.parametrize("body", [
    pytest.param({}, id="empty"),
    pytest.param({"price": None}, id="null-price"),
    pytest.param({"name": " "}, id="blank-name"),
    pytest.param({"seats": 301}, id="over-limit"),
    pytest.param({"unknown": 1}, id="unknown-field"),
])
def test_invalid_patch_preserves_ticket(client, admin_headers, make_item, body):
    ticket = make_item()
    assert client.patch(f"/tickets/{ticket['id']}", json=body, headers=admin_headers).status_code == 422
    assert client.get(f"/tickets/{ticket['id']}", headers=admin_headers).json() == ticket


def test_put_requires_all_fields(client, admin_headers, make_item):
    ticket = make_item()
    response = client.put(f"/tickets/{ticket['id']}", json={"name": "X", "price": 1}, headers=admin_headers)
    assert response.status_code == 422
    assert client.get(f"/tickets/{ticket['id']}", headers=admin_headers).json() == ticket


def test_duplicate_name_returns_409(client, admin_headers, make_item):
    ticket = make_item(name="Unique")
    response = client.post("/tickets", json={"name": "Unique", "price": 1}, headers=admin_headers)
    assert response.status_code == 409
    assert client.get("/tickets", headers=admin_headers).json() == [ticket]


@pytest.mark.parametrize("method", ["PUT", "PATCH"], ids=["put", "patch"])
def test_update_duplicate_name_rolls_back(client, admin_headers, make_item, method):
    make_item(name="Taken")
    ticket = make_item(name="Original")
    response = client.request(method, f"/tickets/{ticket['id']}",
        json={"name": "Taken", "price": 200, "seats": 20}, headers=admin_headers)
    assert response.status_code == 409
    assert client.get(f"/tickets/{ticket['id']}", headers=admin_headers).json() == ticket


def test_seats_increase_returns_new_count(client, admin_headers, make_item):
    ticket = make_item(seats=10)
    response = client.post(f"/tickets/{ticket['id']}/seats", json={"delta": 5}, headers=admin_headers)
    assert response.status_code == 200
    assert response.json() == {**ticket, "seats": 15}


def test_seats_decrease_returns_new_count(client, admin_headers, make_item):
    ticket = make_item(seats=10)
    response = client.post(f"/tickets/{ticket['id']}/seats", json={"delta": -4}, headers=admin_headers)
    assert response.status_code == 200
    assert response.json() == {**ticket, "seats": 6}


@pytest.mark.parametrize("seats,delta", [
    pytest.param(5, -6, id="below-zero"),
    pytest.param(299, 2, id="above-300"),
])
def test_seats_conflict_does_not_change_state(client, admin_headers, make_item, seats, delta):
    ticket = make_item(seats=seats)
    response = client.post(f"/tickets/{ticket['id']}/seats", json={"delta": delta}, headers=admin_headers)
    assert response.status_code == 409
    assert client.get(f"/tickets/{ticket['id']}", headers=admin_headers).json() == ticket


@pytest.mark.parametrize("delta", [0, 1.5, True, "2"], ids=["zero", "float", "bool", "string"])
def test_invalid_delta_returns_422(client, admin_headers, make_item, delta):
    ticket = make_item()
    response = client.post(f"/tickets/{ticket['id']}/seats", json={"delta": delta}, headers=admin_headers)
    assert response.status_code == 422
    assert client.get(f"/tickets/{ticket['id']}", headers=admin_headers).json() == ticket


@pytest.mark.parametrize("seats,delta,expected", [(1, -1, 0), (299, 1, 300)], ids=["zero", "300"])
def test_seats_exact_boundaries_allowed(client, admin_headers, make_item, seats, delta, expected):
    ticket = make_item(seats=seats)
    response = client.post(f"/tickets/{ticket['id']}/seats", json={"delta": delta}, headers=admin_headers)
    assert response.status_code == 200
    assert response.json()["seats"] == expected


def test_name_filter_is_case_insensitive(client, viewer_headers, catalog):
    response = client.get("/tickets", params={"q": "cOnCeRt"}, headers=viewer_headers)
    assert response.status_code == 200
    assert response.json() == [catalog[0], catalog[2]]


def test_name_filter_escapes_wildcards(client, viewer_headers, make_item):
    literal = make_item(name="100%_Ticket")
    make_item(name="Other")
    response = client.get("/tickets", params={"q": "%_"}, headers=viewer_headers)
    assert response.status_code == 200
    assert response.json() == [literal]


def test_name_filter_handles_cyrillic(client, viewer_headers, make_item):
    ticket = make_item(name="Концерт")
    make_item(name="Театр")
    response = client.get("/tickets", params={"q": "КОНЦ"}, headers=viewer_headers)
    assert response.status_code == 200
    assert response.json() == [ticket]


def test_price_filter_includes_boundaries(client, viewer_headers, catalog):
    response = client.get("/tickets", params={"min_price": 100, "max_price": 200}, headers=viewer_headers)
    assert response.status_code == 200
    assert response.json() == catalog[:2]


def test_limit_offset_returns_expected_slice(client, viewer_headers, catalog):
    response = client.get("/tickets", params={"limit": 1, "offset": 1}, headers=viewer_headers)
    assert response.status_code == 200
    assert response.json() == catalog[1:2]


@pytest.mark.parametrize("params", [
    pytest.param({"limit": 0}, id="zero-limit"),
    pytest.param({"limit": 101}, id="large-limit"),
    pytest.param({"offset": -1}, id="negative-offset"),
    pytest.param({"min_price": -1}, id="negative-price"),
    pytest.param({"min_price": 200, "max_price": 100}, id="reversed-range"),
])
def test_invalid_query_returns_422(client, viewer_headers, params):
    assert client.get("/tickets", params=params, headers=viewer_headers).status_code == 422


@pytest.mark.parametrize("currency,expected", [("USD", 1.0), ("EUR", 0.9)], ids=["usd", "eur"])
def test_price_uses_stubbed_rate(client, viewer_headers, make_item, rates, currency, expected):
    ticket = make_item(price=100)
    response = client.get(f"/tickets/{ticket['id']}/price", params={"currency": currency}, headers=viewer_headers)
    assert response.status_code == 200
    assert set(response.json()) == {"currency", "price"}
    assert response.json()["currency"] == currency
    assert response.json()["price"] == pytest.approx(expected)
    rates.rate.assert_called_once_with(currency)


def test_rate_failure_returns_502(client, viewer_headers, make_item, rates):
    ticket = make_item()
    rates.rate.side_effect = RateUnavailable("offline")
    response = client.get(f"/tickets/{ticket['id']}/price?currency=USD", headers=viewer_headers)
    assert response.status_code == 502
    rates.rate.assert_called_once_with("USD")


def test_unknown_currency_returns_422_without_rate_call(client, viewer_headers, make_item, rates):
    ticket = make_item()
    response = client.get(f"/tickets/{ticket['id']}/price?currency=GBP", headers=viewer_headers)
    assert response.status_code == 422
    rates.rate.assert_not_called()


def test_create_after_delete_preserves_other_ticket(client, admin_headers, make_item):
    first = make_item(name="A")
    second = make_item(name="B")
    assert client.delete(f"/tickets/{first['id']}", headers=admin_headers).status_code == 204
    third = make_item(name="C")
    assert third["id"] > second["id"]
    assert client.get(f"/tickets/{second['id']}", headers=admin_headers).json() == second
