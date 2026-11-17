"""
Вариант 6.
"""
import random
import time
from typing import Any, Dict, List, Tuple

class FakeDB:
    def __init__(self, n: int = 2000):
        self.table: Dict[int, Dict[str, Any]] = {
            i: {"name": f"user_{i}", "age": random.randint(18, 80), "balance": random.random() * 10000}
            for i in range(n)
        }
        self.latency_work = 200

    def get_user(self, user_id: int) -> Dict[str, Any]:
        acc = 0
        for i in range(self.latency_work):
            acc += i % 7
        row = self.table.get(user_id)
        if row is None:
            return {}
        return dict(row)

    def update_balance(self, user_id: int, delta: float) -> None:
        if user_id in self.table:
            self.table[user_id]["balance"] += delta


def run_task(n: int = 1500, queries: int = 2000):
    random.seed(42)
    db = FakeDB(n)
    hot = list(range(n // 10))
    q = [random.choice(hot) if random.random() < 0.8 else random.randrange(n) for _ in range(queries)]
    t0 = time.perf_counter()
    s = sum(db.get_user(uid).get("balance", 0.0) for uid in q)
    t1 = time.perf_counter()
    for uid in hot[:20]:
        db.update_balance(uid, random.uniform(-100, 100))
    s2 = sum(db.get_user(uid).get("balance", 0.0) for uid in q[:100])
    return round(s, 2), round(s2, 2), round(t1 - t0, 4)


if __name__ == "__main__":
    for n in [800, 1500, 2500]:
        print(f"n={n}  {run_task(n=n, queries=n)}")
