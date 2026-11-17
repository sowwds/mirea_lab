import random
import pytest
from variant_9_baseline import Analytics, run_task as original_task
from variant_9 import LruAnalytics, ManualAnalytics, run_task


@pytest.mark.parametrize("cls", [LruAnalytics, ManualAnalytics])
def test_reports_and_updates(cls):
    random.seed(42)
    original = Analytics(10, 12)
    random.seed(42)
    cached = cls(10, 12)
    days, products = [0, 1, 0], [0, 2, 0]
    for _ in range(2):
        assert cached.summary_report(days, products) == original.summary_report(days, products)
    for day, product, value in [(0, 0, 100), (0, 2, 0), (5, 7, -3)]:
        original.update_sale(day, product, value)
        cached.update_sale(day, product, value)
        for d in range(10):
            assert cached.daily_total(d) == original.daily_total(d)
        for p in range(12):
            assert cached.product_total(p) == original.product_total(p)
    assert cached.daily_total(-1) == 0
    assert cached.product_total(99) == 0


@pytest.mark.parametrize("cls", [LruAnalytics, ManualAnalytics])
def test_limit_and_eviction(cls):
    random.seed(42)
    obj = cls(10, 12, maxsize=2)
    obj.daily_total(0)
    obj.daily_total(1)
    obj.daily_total(0)  # день 0 снова недавно использован
    obj.daily_total(2)  # должен вытеснить день 1
    if cls is LruAnalytics:
        assert obj._cached.cache_info().currsize == 2
        misses = obj._cached.cache_info().misses
        obj.daily_total(1)
        assert obj._cached.cache_info().misses == misses + 1
    else:
        assert list(obj._cache) == [("day", 0), ("day", 2)]
    assert obj.daily_total(1) == sum(obj.sales[(1, p)] for p in range(12))


def test_selective_invalidation():
    obj = ManualAnalytics(10, 12)
    obj.daily_total(0)
    obj.daily_total(1)
    obj.product_total(2)
    obj.product_total(3)
    obj.update_sale(0, 2, 100)
    assert set(obj._cache) == {("day", 1), ("product", 3)}


def test_full_invalidation():
    obj = LruAnalytics(10, 12)
    obj.daily_total(0)
    obj.product_total(1)
    obj.update_sale(0, 1, 100)
    assert obj._cached.cache_info().currsize == 0


def test_instances_independent():
    a, b = ManualAnalytics(10, 12), ManualAnalytics(10, 12)
    b.daily_total(0)
    expected = b.daily_total(0)
    a.update_sale(0, 0, 999)
    assert b.daily_total(0) == expected


def test_task_result():
    assert run_task(40, 60, 30)[:2] == original_task(40, 60, 30)[:2]
