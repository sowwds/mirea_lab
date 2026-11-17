"""Две ограниченные стратегии кэша для аналитики продаж."""
from collections import OrderedDict
from functools import lru_cache
import random
import time
from variant_9_baseline import Analytics


class LruAnalytics(Analytics):
    """LRU на экземпляр; после изменения данных очищается весь кэш."""
    def __init__(self, n_days=90, n_products=100, maxsize=32):
        super().__init__(n_days, n_products)
        self._cached = lru_cache(maxsize=maxsize)(self._calculate)

    def _calculate(self, kind, key):
        if kind == "day":
            return super().daily_total(key)
        return super().product_total(key)

    def daily_total(self, day):
        return self._cached("day", day)

    def product_total(self, product):
        return self._cached("product", product)

    def update_sale(self, day, product, value):
        super().update_sale(day, product, value)
        self._cached.cache_clear()


class ManualAnalytics(Analytics):
    """LRU на OrderedDict; удаляются только затронутые день и товар."""
    def __init__(self, n_days=90, n_products=100, maxsize=32):
        super().__init__(n_days, n_products)
        if maxsize < 1:
            raise ValueError("maxsize должен быть положительным")
        self.maxsize = maxsize
        self._cache = OrderedDict()

    def _get_total(self, kind, key):
        cache_key = (kind, key)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        value = (super().daily_total(key) if kind == "day"
                 else super().product_total(key))
        self._cache[cache_key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return value

    def daily_total(self, day):
        return self._get_total("day", day)

    def product_total(self, product):
        return self._get_total("product", product)

    def update_sale(self, day, product, value):
        super().update_sale(day, product, value)
        self._cache.pop(("day", day), None)
        self._cache.pop(("product", product), None)


def run_task(n_days=60, n_products=80, reports=40):
    random.seed(42)
    an = ManualAnalytics(n_days, n_products)
    reqs = [(random.sample(range(n_days), k=5),
             random.sample(range(n_products), k=8)) for _ in range(reports)]
    reqs = reqs + reqs[:reports // 2]
    start = time.perf_counter()
    out = [an.summary_report(d, p) for d, p in reqs]
    elapsed = time.perf_counter() - start
    for _ in range(20):
        an.update_sale(random.randrange(n_days), random.randrange(n_products), random.randint(0, 50))
    out2 = [an.summary_report(d, p) for d, p in reqs[:10]]
    return round(sum(x["combo"] for x in out), 2), len(out2), round(elapsed, 4)


if __name__ == "__main__":
    for nd in [40, 60, 80]:
        print(f"days={nd}  {run_task(nd, 60, 30)}")
