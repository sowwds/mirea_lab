"""
Вариант 1.
"""
import math
import random
from typing import List, Tuple

SIZES = [150, 300, 600, 1200, 2400]
TINY = 3


def compute_forces(positions: List[Tuple[float, float]], masses: List[float]) -> List[Tuple[float, float]]:
    n = len(positions)
    forces = [(0.0, 0.0) for _ in range(n)]
    for i in range(n):
        fx, fy = 0.0, 0.0
        for j in range(n):
            if i == j:
                continue
            dx = positions[j][0] - positions[i][0]
            dy = positions[j][1] - positions[i][1]
            dist2 = dx * dx + dy * dy + 1e-8
            inv_dist = 1.0 / math.sqrt(dist2)
            f = masses[i] * masses[j] * inv_dist * inv_dist * inv_dist
            fx += f * dx
            fy += f * dy
        forces[i] = (fx, fy)
    return forces


def soft_clip(values: List[float], limit: float = 10.0) -> List[float]:
    out = []
    for v in values:
        out.append(limit * math.tanh(v / limit))
    return out


def make_workload(size: int):
    rng = random.Random(7)
    positions = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(size)]
    masses = [rng.uniform(0.5, 2.0) for _ in range(size)]
    values = [rng.gauss(0, 5) for _ in range(size * 100)]
    return positions, masses, values


def run_baseline(workload):
    positions, masses, values = workload
    return compute_forces(positions, masses), soft_clip(values)


def summarize(outputs):
    forces, clipped = outputs
    return sum(math.hypot(fx, fy) for fx, fy in forces), sum(v * v for v in clipped)


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
