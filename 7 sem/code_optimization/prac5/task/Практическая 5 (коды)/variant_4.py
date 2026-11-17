"""
Вариант 4.
"""
import random
import time
from typing import Dict

class CombinatorialService:
    def __init__(self):
        self.modifier = 1.0

    def compute(self, n: int, k: int) -> float:
        if k < 0 or k > n:
            return 0.0
        if k == 0 or k == n:
            return self.modifier
        return self.compute(n - 1, k - 1) + self.compute(n - 1, k) * self.modifier

    def set_modifier(self, value: float) -> None:
        self.modifier = value


def run_task(max_n: int = 18, queries: int = 80):
    random.seed(42)
    svc = CombinatorialService()
    pairs = [(random.randint(10, max_n), random.randint(2, 6)) for _ in range(queries)]
    pairs = pairs + pairs[: queries // 2]
    t0 = time.perf_counter()
    s = sum(svc.compute(n, k) for n, k in pairs)
    t1 = time.perf_counter()
    svc.set_modifier(1.1)
    s2 = sum(svc.compute(n, k) for n, k in pairs[:20])
    return round(s, 2), round(s2, 2), round(t1 - t0, 4)


if __name__ == "__main__":
    for mn in [14, 16, 18]:
        print(f"max_n={mn}  {run_task(max_n=mn, queries=40)}")
