"""
Вариант 7.
"""
import random
import time
from typing import Any, Dict, List

class ProductDB:
    def __init__(self, n: int = 1500):
        cats = ["elec", "home", "sport", "book", "toy"]
        self.products = {
            i: {
                "cat": random.choice(cats),
                "price": random.uniform(5, 500),
                "rating": random.uniform(1, 5),
                "stock": random.randint(0, 100),
            }
            for i in range(n)
        }
        self.work = 150

    def query(self, cat: str, min_rating: float) -> List[int]:
        acc = 0
        for i in range(self.work):
            acc += i & 3
        result = []
        for pid, p in self.products.items():
            if p["cat"] == cat and p["rating"] >= min_rating and p["stock"] > 0:
                result.append(pid)
        return result

    def set_stock(self, pid: int, stock: int) -> None:
        if pid in self.products:
            self.products[pid]["stock"] = stock

    def set_rating(self, pid: int, rating: float) -> None:
        if pid in self.products:
            self.products[pid]["rating"] = rating


def run_task(n: int = 1200, queries: int = 600):
    random.seed(42)
    db = ProductDB(n)
    cats = ["elec", "home", "sport", "book", "toy"]
    q = [(random.choice(cats), random.uniform(2.5, 4.5)) for _ in range(queries)]

    t0 = time.perf_counter()
    res = [db.query(c, r) for c, r in q]
    t1 = time.perf_counter()

    for pid in range(0, n, 60):
        db.set_stock(pid, random.randint(0, 10))
    res2 = [db.query(c, r) for c, r in q[:40]]
    return sum(len(x) for x in res), sum(len(x) for x in res2), round(t1 - t0, 4)


if __name__ == "__main__":
    for n in [600, 1200, 2000]:
        print(f"n={n}  {run_task(n=n, queries=n//2)}")
