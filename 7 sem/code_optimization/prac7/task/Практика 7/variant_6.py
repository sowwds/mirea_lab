"""
Вариант 6.
"""
import math
import random
from array import array
from typing import Tuple

N_STEPS = 200
SIZES = [400, 800, 1600, 3200, 6400]
TINY = 1


def monte_carlo_pi(xs, ys) -> float:
    inside = 0
    n = len(xs)
    for i in range(n):
        x = xs[i]
        y = ys[i]
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / n


def estimate_integral(us) -> float:
    s = 0.0
    for x in us:
        s += math.exp(-x * x)
    return s / len(us)


def random_walk_stats(steps, n_walks: int, n_steps: int) -> Tuple[float, float]:
    final_positions = []
    for k in range(n_walks):
        pos = 0.0
        base = k * n_steps
        for t in range(n_steps):
            pos += steps[base + t]
        final_positions.append(pos)
    mean = sum(final_positions) / n_walks
    var = sum((p - mean) ** 2 for p in final_positions) / n_walks
    return mean, math.sqrt(var)


def make_workload(size: int):
    rng = random.Random(23)
    n_pi, n_int, n_walks = size * 1000, size * 500, max(1, size)
    xs = array('d', (rng.random() for _ in range(n_pi)))
    ys = array('d', (rng.random() for _ in range(n_pi)))
    us = array('d', (rng.random() for _ in range(n_int)))
    steps = array('b', (1 if rng.random() < 0.5 else -1 for _ in range(n_walks * N_STEPS)))
    return xs, ys, us, steps, n_walks, N_STEPS


def run_baseline(workload):
    xs, ys, us, steps, n_walks, n_steps = workload
    return monte_carlo_pi(xs, ys), estimate_integral(us), random_walk_stats(steps, n_walks, n_steps)


def summarize(outputs):
    pi_est, integ, (mean, std) = outputs
    return pi_est, integ, mean, std


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
