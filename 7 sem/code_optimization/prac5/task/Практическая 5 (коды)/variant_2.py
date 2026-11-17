"""
Вариант 2.
"""
import random
import time
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Record:
    id: int
    group: str
    value: float
    active: bool

class RecordStore:
    def __init__(self, n: int = 2000):
        groups = ["A", "B", "C", "D", "E"]
        self.records: List[Record] = [
            Record(i, random.choice(groups), random.random() * 100, random.random() > 0.2)
            for i in range(n)
        ]

    def group_stats(self, group: str) -> Dict[str, float]:
        vals = [r.value for r in self.records if r.group == group and r.active]
        if not vals:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "max": 0.0}
        s = 0.0
        mx = vals[0]
        for v in vals:
            s += v
            if v > mx:
                mx = v
            for _ in range(15):
                s += 0.0
        return {"count": len(vals), "sum": s, "avg": s / len(vals), "max": mx}

    def update_record(self, rid: int, value: float = None, active: bool = None) -> None:
        for r in self.records:
            if r.id == rid:
                if value is not None:
                    r.value = value
                if active is not None:
                    r.active = active
                break

    def add_record(self, group: str, value: float) -> None:
        nid = len(self.records)
        self.records.append(Record(nid, group, value, True))


def run_task(n: int = 1500, queries: int = 800):
    random.seed(42)
    store = RecordStore(n)
    groups = ["A", "B", "C", "D", "E"]
    q = [random.choice(groups) for _ in range(queries)]

    t0 = time.perf_counter()
    results = [store.group_stats(g) for g in q]
    t1 = time.perf_counter()

    for i in range(0, n, 100):
        store.update_record(i, value=random.random() * 100)
    results2 = [store.group_stats(g) for g in q[:50]]
    total = sum(r["sum"] for r in results)
    return round(total, 2), len(results2), round(t1 - t0, 4)


if __name__ == "__main__":
    for n in [800, 1500, 2500]:
        print(f"n={n}  {run_task(n=n, queries=n//2)}")
