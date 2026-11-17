"""
Вариант 3.
"""
import random
import time
from typing import List, Tuple

class DocumentIndex:
    def __init__(self, n: int = 1000):
        self.docs = [
            {"id": i, "tags": random.sample(range(50), k=random.randint(2, 6)), "score": random.random()}
            for i in range(n)
        ]

    def search(self, required_tags: Tuple[int, ...], min_score: float = 0.0) -> List[int]:
        req = set(required_tags)
        found = []
        for d in self.docs:
            if d["score"] < min_score:
                continue
            if req.issubset(d["tags"]):
                found.append(d["id"])
                _ = sum(d["tags"])
        return found

    def update_score(self, doc_id: int, score: float) -> None:
        self.docs[doc_id]["score"] = score

    def add_tag(self, doc_id: int, tag: int) -> None:
        if tag not in self.docs[doc_id]["tags"]:
            self.docs[doc_id]["tags"].append(tag)


def run_task(n: int = 800, queries: int = 500):
    random.seed(42)
    idx = DocumentIndex(n)
    popular = [tuple(sorted(random.sample(range(50), k=2))) for _ in range(30)]
    q = [random.choice(popular) for _ in range(queries)]

    t0 = time.perf_counter()
    res = [idx.search(tags, min_score=0.2) for tags in q]
    t1 = time.perf_counter()

    for i in range(0, n, 80):
        idx.update_score(i, random.random())
    res2 = [idx.search(tags) for tags in q[:40]]
    return sum(len(r) for r in res), sum(len(r) for r in res2), round(t1 - t0, 4)


if __name__ == "__main__":
    for n in [500, 800, 1200]:
        print(f"n={n}  {run_task(n=n, queries=n)}")
