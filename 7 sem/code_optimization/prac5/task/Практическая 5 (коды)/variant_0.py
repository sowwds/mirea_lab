"""
Вариант 0.
"""
import random
import time
from typing import Dict, List, Tuple

class ScoreService:
    def __init__(self, n_items: int = 500):
        self.weights: Dict[int, float] = {i: random.random() for i in range(n_items)}
        self._factor = 1.17

    def expensive_score(self, item_id: int, depth: int = 80) -> float:
        if item_id not in self.weights:
            return 0.0
        base = self.weights[item_id]
        acc = 0.0
        x = base
        for i in range(depth):
            x = (x * self._factor + 0.001 * i) % 1.0
            acc += x * base
        return acc

    def update_weight(self, item_id: int, new_weight: float) -> None:
        self.weights[item_id] = new_weight

    def batch_scores(self, ids: List[int]) -> List[float]:
        return [self.expensive_score(i) for i in ids]


def make_workload(service: ScoreService, queries: int = 2000, unique_ratio: float = 0.25):
    n = len(service.weights)
    hot = list(range(0, max(1, int(n * unique_ratio))))
    ids = [random.choice(hot) if random.random() < 0.7 else random.randrange(n) for _ in range(queries)]
    return ids


def run_task(n_items: int = 400, queries: int = 1500):
    random.seed(42)
    svc = ScoreService(n_items)
    ids = make_workload(svc, queries=queries)
    t0 = time.perf_counter()
    scores = svc.batch_scores(ids)
    t1 = time.perf_counter()
    for i in range(0, n_items, 50):
        svc.update_weight(i, random.random())
    scores2 = svc.batch_scores(ids[:200])
    return round(sum(scores), 4), round(sum(scores2), 4), round(t1 - t0, 4)


if __name__ == "__main__":
    for n in [200, 400, 600]:
        res = run_task(n_items=n, queries=n * 3)
        print(f"n_items={n}  result={res}")
