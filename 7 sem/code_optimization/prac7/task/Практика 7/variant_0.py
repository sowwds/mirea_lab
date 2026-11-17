"""
Вариант 0.
"""
import math
import random
from typing import List, Tuple

SIZES = [250, 500, 1000, 2000, 4000]
TINY = 3


def pairwise_distances(points: List[Tuple[float, float]]) -> List[float]:
    n = len(points)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dists.append(math.sqrt(dx * dx + dy * dy))
    return dists


def poly_eval(coeffs: List[float], xs: List[float]) -> List[float]:
    results = []
    for x in xs:
        val = 0.0
        for c in reversed(coeffs):
            val = val * x + c
        results.append(val)
    return results


def make_workload(size: int):
    rng = random.Random(42)
    points = [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(size)]
    coeffs = [rng.uniform(-5, 5) for _ in range(8)]
    xs = [rng.uniform(-10, 10) for _ in range(size * 20)]
    return points, coeffs, xs


def run_baseline(workload):
    points, coeffs, xs = workload
    return pairwise_distances(points), poly_eval(coeffs, xs)


def summarize(outputs):
    d, p = outputs
    mean_d = sum(d) / len(d) if d else 0.0
    mean_p2 = sum(v * v for v in p) / len(p) if p else 0.0
    return len(d), mean_d, mean_p2


def run_task(size: int):
    import time
    workload = make_workload(size)
    t0 = time.perf_counter()
    outputs = run_baseline(workload)
    elapsed = time.perf_counter() - t0
    return summarize(outputs), elapsed


if __name__ == "__main__":
    for s in SIZES:
        res, t = run_task(s)
        print(f"size={s:<6} {res}  time={t:.4f}s")
