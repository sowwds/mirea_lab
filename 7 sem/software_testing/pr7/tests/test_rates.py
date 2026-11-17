"""Адаптер курсов проверяется через MockTransport без внешней сети."""

import httpx
import pytest

from app.rates import ExchangeRates, RateUnavailable, get_rates


def test_exchange_adapter_reads_rub_rate():
    def handler(request):
        assert str(request.url) == "https://open.er-api.com/v6/latest/RUB"
        return httpx.Response(200, json={"result": "success", "rates": {"USD": 0.01}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert ExchangeRates(client).rate("USD") == 0.01


@pytest.mark.parametrize("data", [
    pytest.param({"result": "error"}, id="provider-error"),
    pytest.param({}, id="missing-fields"),
    pytest.param({"result": "success", "rates": {"USD": "bad"}}, id="string-rate"),
    pytest.param({"result": "success", "rates": {"USD": 0}}, id="zero-rate"),
    pytest.param({"result": "success", "rates": {"USD": True}}, id="bool-rate"),
    pytest.param(None, id="null-response"),
])
def test_exchange_adapter_rejects_invalid_data(data):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=data))) as client:
        with pytest.raises(RateUnavailable):
            ExchangeRates(client).rate("USD")


def test_exchange_adapter_wraps_timeout():
    def handler(request):
        raise httpx.ConnectTimeout("offline")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RateUnavailable):
            ExchangeRates(client).rate("USD")


def test_rates_dependency_closes_client():
    dependency = get_rates()
    service = next(dependency)
    assert not service.client.is_closed
    dependency.close()
    assert service.client.is_closed
