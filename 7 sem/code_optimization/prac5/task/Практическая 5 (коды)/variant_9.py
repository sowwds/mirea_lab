"""
Вариант 9.
"""
import random
import time
from typing import Dict, List

class Analytics:
    def __init__(self, n_days: int = 90, n_products: int = 100):
        self.n_days = n_days
        self.n_products = n_products
        self.sales = {
            (d, p): random.randint(0, 50)
            for d in range(n_days)
            for p in range(n_products)
        }

    def daily_total(self, day: int) -> float:
        s = 0.0
        for p in range(self.n_products):
            s += self.sales.get((day, p), 0)
            for _ in range(10):
                s += 0.0
        return s

    def product_total(self, product: int) -> float:
        s = 0.0
        for d in range(self.n_days):
            s += self.sales.get((d, product), 0)
            for _ in range(10):
                s += 0.0
        return s

    def summary_report(self, days: List[int], products: List[int]) -> Dict[str, float]:
        dt = sum(self.daily_total(d) for d in days)
        pt = sum(self.product_total(p) for p in products)
        return {"daily_sum": dt, "product_sum": pt, "combo": dt + pt * 0.01}

    def update_sale(self, day: int, product: int, value: int) -> None:
        self.sales[(day, product)] = value


def run_task(n_days: int = 60, n_products: int = 80, reports: int = 40):
    random.seed(42)
    an = Analytics(n_days, n_products)
    reqs = []
    for _ in range(reports):
        days = random.sample(range(n_days), k=5)
        products = random.sample(range(n_products), k=8)
        reqs.append((days, products))
    reqs = reqs + reqs[: reports // 2]
    t0 = time.perf_counter()
    out = [an.summary_report(d, p) for d, p in reqs]
    t1 = time.perf_counter()
    for _ in range(20):
        an.update_sale(random.randrange(n_days), random.randrange(n_products), random.randint(0, 50))
    out2 = [an.summary_report(d, p) for d, p in reqs[:10]]
    return round(sum(x["combo"] for x in out), 2), len(out2), round(t1 - t0, 4)


if __name__ == "__main__":
    for nd in [40, 60, 80]:
        print(f"days={nd}  {run_task(n_days=nd, n_products=60, reports=30)}")
