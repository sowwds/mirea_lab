"""
Вариант 1.
"""
import random
import time
from typing import Dict, List

class PricingEngine:
    def __init__(self, n: int = 300):
        self.base_price = {i: 10 + random.random() * 90 for i in range(n)}
        self.discount = {i: random.uniform(0.0, 0.3) for i in range(n)}

    def final_price(self, item_id: int, qty: int) -> float:
        if item_id not in self.base_price:
            return 0.0
        p = self.base_price[item_id]
        d = self.discount[item_id]
        total = 0.0
        for q in range(1, qty + 1):
            total += p * (1.0 - d) * (1.0 + 0.0001 * q)
            for _ in range(30):
                total = total * 1.000001 + 0.0
        return total

    def set_discount(self, item_id: int, value: float) -> None:
        self.discount[item_id] = value

    def set_base(self, item_id: int, value: float) -> None:
        self.base_price[item_id] = value


def run_task(n: int = 300, requests: int = 1000):
    random.seed(42)
    eng = PricingEngine(n)
    reqs = [(random.randrange(n), random.randint(1, 20)) for _ in range(requests)]
    for i in range(0, requests, 5):
        reqs[i] = reqs[i % 50]

    t0 = time.perf_counter()
    s = sum(eng.final_price(i, q) for i, q in reqs)
    t1 = time.perf_counter()

    for i in range(0, n, 40):
        eng.set_discount(i, random.uniform(0, 0.4))
    s2 = sum(eng.final_price(i, q) for i, q in reqs[:100])
    return round(s, 2), round(s2, 2), round(t1 - t0, 4)


if __name__ == "__main__":
    for n in [150, 300, 450]:
        print(f"n={n}  {run_task(n=n, requests=n*3)}")
